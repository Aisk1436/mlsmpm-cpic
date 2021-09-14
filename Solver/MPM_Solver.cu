#include "utils.h"
#include "MPM_Solver.cuh"

namespace Mpm
{
    constexpr int thread_per_block = 256;

    void cuda_check_error()
    {
        cudaDeviceSynchronize();
        auto status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            utils::error(cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    constexpr Real E = 5e3;
    constexpr Real nu = 0.3; // 0.2;

    __constant__ Real w_k = 50.0;
    __constant__ Real w_gamma = 3.0;
    __constant__ Real mu0 = E / (2 * (1 + nu));
    __constant__ Real lambda0 = E * nu / ((1 + nu) * (1 - 2 * nu));

    __device__ Vec
    velocity_sub_projection(const Vec& v, const Vec& n, SurfaceType b, Real mu);

    __device__ Vec
    velocity_projection(const Rigid_body& rb, const Particle& p, const Vec& x);

    __device__ Mat get_rotation_matrix(Vec v);

    __device__ void
    apply_impulse(Rigid_body& rb, const Vec& impulse, const Vec& grid_pos);

    __device__ inline bool judge_binary(int a, int b)
    {
        return (a & (1 << b)) == (1 << b);
    }

    __global__ void grid_init_kernel(Grid* grid)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        grid[idx].init();
    }

    __global__ void read_cdf_line_kernel
            (Grid* grid, Parameter* parameter,
                    Rigid_Particle* rigid_particles)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_rigid_particles) return;
        auto& rp = rigid_particles[threadId];
        auto& pa = *parameter;
        auto base = (rp.x * pa.inv_dx - Real(0.5)).cast<int>();
        auto rm = get_rotation_matrix(rp.uaxis);

        for (auto i = 0; i < 3; i++)
            for (auto j = 0; j < 3; j++)
            {
                auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                auto grid_pos =
                        Veci(base[0] + i, base[1] + j).cast<Real>() * pa.dx;
                auto rotated = rm % (grid_pos - rp.x);
                rotated[0] *= pa.inv_dx;
                if (rotated[0] < 0.02 || 1.02 < rotated[0]) continue;
                auto negative = rotated[1] < 0;
#if __CUDA_ARCH__ >= 600
                bool isSet;
                do
                {
                    if ((isSet = atomicCAS(&grid[idx].mutex, 0, 1)))
                    {
                        grid[idx].A_states |= 1 << rp.id;
                        grid[idx].T_states |= int(negative) << rp.id;
                        if (grid[idx].closest_rigid_body == -1 ||
                            abs(rotated[1]) < grid[idx].d)
                        {
                            grid[idx].closest_rigid_body = rp.id;
                            grid[idx].d = (abs(rotated[1]));
                        }
                    }
                    if (isSet) grid[idx].mutex = false;
                } while (!isSet);
#endif
            }
    }

    __global__ void read_cdf_circle_kernel
            (Grid* grid, Parameter* parameter, Rigid_body* rigid_bodies)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        auto& pa = *parameter;
        if (idx >= pa.n_grid * pa.n_grid) return;
        auto& g = grid[idx];
        auto grid_pos =
                Veci(utils::narrow_cast<int>(idx) / pa.n_grid,
                        utils::narrow_cast<int>(idx) % pa.n_grid).cast<Real>() *
                pa.dx;
        for (auto r = 0; r < pa.n_rigid_bodies; r++)
        {
            if (rigid_bodies[r].bt != BodyType::circle) continue;
            auto& rb = rigid_bodies[r];
            auto L = (grid_pos - rb.x).norm();
            auto dist = L - rb.r;
            if (abs(dist) > 2.83 * pa.dx) continue;
            auto negative = dist < 0;
            g.A_states |= 1 << r;
            g.T_states |= int(negative) << r;
            if (g.closest_rigid_body == -1 || abs(dist) < g.d)
            {
                g.closest_rigid_body = r;
                g.d = dist;
            }
        }
    }

    __global__ void cdf_reconstruction_kernel
            (class Particle* particles, Grid* grid, Parameter* parameter)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_particles) return;
        auto& p = particles[threadId];
        auto& pa = *parameter;
        if (!(0.0 < p.x[0] && p.x[0] < 1.0
              && 0.0 < p.x[1] && p.x[1] < 1.0))
            return;
        auto base = (p.x * pa.inv_dx - Real(0.5)).cast<int>();
        auto fx = p.x * pa.inv_dx - base.cast<Real>();
        Vec w[3] = {
                (fx - Real(1.5)) * (fx - Real(1.5)) * Real(0.5),
                (fx - Real(1.0)) * (fx - Real(1.0)) * Real(-1.0) + Real(0.75),
                (fx - Real(0.5)) * (fx - Real(0.5)) * Real(0.5),
        };

        auto boundaries = 0;
        for (auto i = 0; i < 3; i++)
            for (auto j = 0; j < 3; j++)
            {
                auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                boundaries |= grid[idx].A_states;
            }

        p.A_states &= boundaries;
        p.T_states &= boundaries;
        boundaries &= (~p.A_states);
        for (auto r = 0; r < pa.n_rigid_bodies; r++)
        {
            if (!judge_binary(boundaries, r)) continue;
            Real d_tmp[2] = { 0.0, 0.0 };
            for (auto i = 0; i < 3; i++)
                for (auto j = 0; j < 3; j++)
                {
                    auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                    if (!judge_binary(grid[idx].A_states, r)) continue;
                    p.A_states |= 1 << r;
                    auto weight = w[i][0] * w[j][1];
                    d_tmp[int(judge_binary(grid[idx].T_states, r))] +=
                            grid[idx].d * weight;
                }
            if (d_tmp[0] + d_tmp[1] > 1e-7)
                p.T_states |= int(d_tmp[1] > d_tmp[0]) << r;
        }

        if (p.A_states)
        {
            Mat3d M;
            Vec3d Qu;
            for (auto i = 0; i < 3; i++)
                for (auto j = 0; j < 3; j++)
                {
                    auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                    if (!grid[idx].A_states) continue;
                    auto dpos = (fx - Vec(utils::narrow_cast<Real>(i),
                            utils::narrow_cast<Real>(j)));
                    auto d = grid[idx].d * pa.inv_dx;
                    Vec3d xp((dpos * Real(-1.0)), 1);
                    auto weight = w[i][0] * w[j][1];
                    auto mask = (p.A_states & grid[idx].A_states);
                    auto diff =
                            (p.T_states & mask) ^ (grid[idx].T_states & mask);
                    if (!diff)
                    {
                        M += xp.outer(xp) * weight;
                        Qu += Vec3d((dpos * -d), d) * weight;
                    }
                    else if (diff > 0 && !(diff & diff - 1))
                    {
                        M += xp.outer(xp) * weight;
                        Qu += Vec3d((dpos * d), -d) * weight;
                    }
                }

            if (abs(M.det()) > 3e-3)
            {
                auto r = M.inv() % Qu;
                p.d = r[2] * pa.dx;
                Vec gra_d(r[0], r[1]);
                if (gra_d.norm() > 1e-2)
                {
                    p.normal = gra_d * Real(1.0 / gra_d.norm());
                }
                else
                {
                    p.normal = Vec(0);
                }
            }
            else
            {
                p.A_states = p.T_states = 0;
                p.d = 0;
                p.normal = Vec(0);
            }
        }
    }

    __global__ void particle_update_kernel
            (class Particle* particles, Parameter* parameter)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_particles) return;
        auto& p = particles[threadId];
        auto& pa = *parameter;
        if (!(0.0 < p.x[0] && p.x[0] < 1.0
              && 0.0 < p.x[1] && p.x[1] < 1.0))
            return;
        p.F = (Mat(1.0) + p.C * pa.dt) % p.F;
        auto[U, sig, Vt] = p.F.svd_solve();
        Mat stress;
        if (p.type == Type::liquid)
        {
            stress = Mat{ Real(w_k * (1.0 - 1.0 / pow(p.J, w_gamma) * p.J)) };
        }
        else
        {
            auto h = max(0.1f, min(5.0f, exp(10 * (1.0f - p.Jp))));
            if (p.type == Type::elastic) h = p.h;
            auto mu = mu0 * h;
            auto la = lambda0 * h;
            auto J = Real(1.0);
            for (int i = 0; i < 2; i++)
            {
                auto new_sig = sig(i, i);
                if (p.type == Type::snow)
                    new_sig = min(max(sig(i, i), Real(1 - 2.5e-2)),
                            Real(1 + 4.5e-3));
                p.Jp *= sig(i, i) / new_sig;
                sig(i, i) = new_sig;
                J *= new_sig;
            }
            if (p.type == Type::snow) p.F = U % sig % Vt;
            stress = (p.F - U % Vt) % p.F.tran() * mu * Real(2.0)
                     + Mat(1.0) * la * J * (J - 1);
        }
        stress = stress * -pa.p_vol * 4.0 * pa.inv_dx * pa.inv_dx * pa.dt;
        p.affine = stress + p.C * pa.p_mass;
        //p.v += pa.gravity * pa.dt;
    }

    __global__ void particle_to_grid_kernel
            (class Particle* particles, Grid* grid,
                    Parameter* parameter, Rigid_body* rigid_bodies)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_particles) return;
        auto& p = particles[threadId];
        auto& pa = *parameter;
        if (!(0.0 < p.x[0] && p.x[0] < 1.0
              && 0.0 < p.x[1] && p.x[1] < 1.0))
            return;
        auto base = (p.x * pa.inv_dx - Real(0.5)).cast<int>();
        auto fx = p.x * pa.inv_dx - base.cast<Real>();
        Vec w[3] = {
                (fx - Real(1.5)) * (fx - Real(1.5)) * Real(0.5),
                (fx - Real(1.0)) * (fx - Real(1.0)) * Real(-1.0) + Real(0.75),
                (fx - Real(0.5)) * (fx - Real(0.5)) * Real(0.5),
        };
        for (auto i = 0; i < 3; i++)
            for (auto j = 0; j < 3; j++)
            {
                auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                Real weight = w[i][0] * w[j][1];
                auto mask = grid[idx].A_states & p.A_states;
                auto diff = (grid[idx].T_states & mask)
                            ^ (p.T_states & mask);
                if (diff == 0)
                {
                    auto dpos = (Vec(utils::narrow_cast<Real>(i),
                            utils::narrow_cast<Real>(j)) - fx) * pa.dx;
                    auto grid_v_add =
                            (p.v * pa.p_mass + p.affine % dpos) * weight;
#if __CUDA_ARCH__ >= 600
                    atomicAdd(&(grid[idx].v[0]), grid_v_add[0]);
                    atomicAdd(&(grid[idx].v[1]), grid_v_add[1]);
                    atomicAdd(&grid[idx].m, weight * pa.p_mass);
#endif
                }
                else
                {
                    auto grid_pos =
                            Veci(base[0] + i, base[1] + j).cast<Real>() * pa.dx;
                    auto& rb = rigid_bodies[grid[idx].closest_rigid_body];
                    auto delta_v = p.v - velocity_projection(rb, p, grid_pos);
                    auto impulse = delta_v * pa.p_mass * weight;
                    apply_impulse(rb, impulse, grid_pos);
                }
            }
    }

    __global__ void grid_update_kernel
            (Grid* grid, Parameter* parameter)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid[idx].m <= 0.0) return;
        auto& pa = *parameter;
        auto& v = grid[idx].v;
        auto i = idx / pa.n_grid;
        auto j = idx % pa.n_grid;
        v = v * Real(1.0 / grid[idx].m);
        v += pa.gravity * pa.dt;
        if (i < 3 && v[0] < 0)
        { v[0] = 0.0; }
        if (i > pa.n_grid - 3 && v[0] > 0)
        { v[0] = 0.0; }
        if (j < 3 && v[1] < 0)
        { v[1] = 0.0; }
        if (j > pa.n_grid - 3 && v[1] > 0)
        { v[1] = 0.0; }
    }

    __global__ void grid_to_particle_kernel
            (class Particle* particles, Grid* grid,
                    Parameter* parameter, Rigid_body* rigid_bodies)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_particles) return;
        auto& p = particles[threadId];
        auto& pa = *parameter;
        if (!(0.0 < p.x[0] && p.x[0] < 1.0
              && 0.0 < p.x[1] && p.x[1] < 1.0))
            return;
        auto base = (p.x * pa.inv_dx - Real(0.5)).cast<int>();
        auto fx = p.x * pa.inv_dx - base.cast<Real>();
        Vec w[3] = {
                (fx - Real(1.5)) * (fx - Real(1.5)) * Real(0.5),
                (fx - Real(1.0)) * (fx - Real(1.0)) * Real(-1.0) + Real(0.75),
                (fx - Real(0.5)) * (fx - Real(0.5)) * Real(0.5),
        };

        int rigid_id = -1;
        Vec new_v{ 0.0, 0.0 };
        Mat new_C{ 0.0, 0.0, 0.0, 0.0 };
        for (auto i = 0; i < 3; i++)
            for (auto j = 0; j < 3; j++)
            {
                auto idx = (base[0] + i) * pa.n_grid + base[1] + j;
                Real weight;
                auto mask = grid[idx].A_states & p.A_states;
                auto diff = (grid[idx].T_states & mask)
                            ^ (p.T_states & mask);
                auto dpos = Vec(utils::narrow_cast<Real>(i),
                        utils::narrow_cast<Real>(j)) - fx;
                weight = w[i][0] * w[j][1];
                Vec v;
                if (diff == 0)
                {
                    v = grid[(base[0] + i) * pa.n_grid + base[1] + j].v;
                }
                else if (grid[idx].closest_rigid_body > -1)
                {
                    auto grid_pos =
                            Veci(base[0] + i, base[1] + j).cast<Real>() * pa.dx;
                    rigid_id = grid[idx].closest_rigid_body;
                    Rigid_body& rb = rigid_bodies[rigid_id];
                    v = velocity_projection(rb, p, grid_pos);
                }
                new_v = new_v + v * weight;
                new_C = new_C + v.outer(dpos) * Real(4.0) * pa.inv_dx * weight;
            }

        p.v = new_v;
        p.C = new_C;
        p.J = (1 + pa.dt * p.C.tr()) * p.J;
        p.x = p.x + p.v * pa.dt;
        if (-0.3 * pa.dx < p.d && p.d < -0.05 * pa.dx)
        {
            Vec dv = p.normal * Real(-p.d * 3e2);
            p.v += dv;
            if (rigid_id != -1)
                apply_impulse(rigid_bodies[rigid_id],
                        dv * -pa.p_mass, p.x);
        }

        /*    // debug
        if (!(0.0 < p.x[0] && p.x[0] < 1.0
            && 0.0 < p.x[1] && p.x[1] < 1.0)) {
            int a = 0;
        }*/
    }

    __global__ void rigid_pre_advection_kernel
            (Parameter* parameter, Rigid_body* rigid_bodies)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_rigid_bodies) return;
        auto& rb = rigid_bodies[threadId];
        auto& pa = *parameter;
        rb.w = 6.0;
        rb.theta += rb.w * pa.dt;
    }

    __global__ void rigid_advection_kernel
            (Parameter* parameter, Rigid_body* rigid_bodies,
                    Rigid_Particle* rigid_particles)
    {
        auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId >= parameter->n_rigid_particles) return;
        auto& rp = rigid_particles[threadId];
        auto& pa = *parameter;
        auto& rb = rigid_bodies[rp.id];
        auto r_pos = rp.x - rb.x;
        if (r_pos.norm() < 1e-10) return;
        auto theta_r = atan2(r_pos[1], r_pos[0]);
        auto theta_axis = atan2(rp.uaxis[1], rp.uaxis[0]);
        theta_r += rb.w * pa.dt;
        theta_axis += rb.w * pa.dt;
        rp.x = Vec(rb.x[0] + cos(theta_r) * r_pos.norm(),
                rb.x[1] + sin(theta_r) * r_pos.norm());
        rp.uaxis = Vec(cos(theta_axis), sin(theta_axis));
    }

    Solver::Solver(int n_p, int n_g, Real dt)
    {
        reset(n_p, n_g, dt);
    }

    void Solver::init()
    {
        pa.n_particles = 0;
        pa.n_rigid_particles = 0;
        pa.n_rigid_bodies = 0;
        pa.dx = 1.0f / utils::narrow_cast<Real>(pa.n_grid);
        pa.inv_dx = utils::narrow_cast<Real>(pa.n_grid);
        pa.gravity[0] = 0.0;
        pa.gravity[1] = -30.0;
        particles = std::make_unique<Particle[]>(m_particles);
        rigid_particles = std::make_unique<Rigid_Particle[]>(m_rigid_particles);
        pa.p_vol = (pa.dx * 0.5f) * (pa.dx * 0.5f);
        pa.p_rho = 1.0;
        pa.p_mass = pa.p_vol * pa.p_rho;

        cudaMalloc(&dev_p, m_particles * sizeof(Particle));
        cudaMalloc(&dev_grid, pa.n_grid * pa.n_grid * sizeof(Grid));
        cudaMalloc(&dev_para, sizeof(Parameter));
        cudaMalloc(&dev_rigid_bodies, 32 * sizeof(Rigid_body));
        cudaMalloc(&dev_rigid_particles,
                m_rigid_particles * sizeof(Rigid_Particle));
        cuda_check_error();
    }

    void Solver::reset(int n_p, int n_g, Real t)
    {
        m_particles = n_p;
        m_rigid_particles = n_g * n_g;
        pa.n_grid = n_g;
        pa.dt = t;
        pa.n_particles = 0;
        pa.n_rigid_particles = 0;
        cudaFree(dev_p);
        cudaFree(dev_grid);
        cudaFree(dev_para);
        cudaFree(dev_rigid_bodies);
        cudaFree(dev_rigid_particles);
        init();
    }

    void Solver::advance(int n_step)
    {
        auto block_particles =
                (pa.n_particles + thread_per_block - 1) / thread_per_block;
        auto block_grid = (pa.n_grid * pa.n_grid + thread_per_block - 1) /
                          thread_per_block;
        auto block_rigid_particles =
                (pa.n_rigid_particles + thread_per_block - 1) /
                thread_per_block;

        while (n_step--)
        {
            if (block_grid)
            {
                grid_init_kernel <<<block_grid, thread_per_block>>>(
                        dev_grid);
            }

            cuda_check_error();

            if (block_rigid_particles)
            {
                read_cdf_line_kernel <<<block_rigid_particles, thread_per_block>>>
                        (dev_grid, dev_para, dev_rigid_particles);
            }

            if (block_grid)
            {
                read_cdf_circle_kernel <<<block_grid, thread_per_block>>>
                        (dev_grid, dev_para, dev_rigid_bodies);
            }

            if (block_particles)
            {
                cdf_reconstruction_kernel <<<block_particles, thread_per_block>>>
                        (dev_p, dev_grid, dev_para);

                particle_update_kernel <<<block_particles, thread_per_block>>>
                        (dev_p, dev_para);

                particle_to_grid_kernel <<<block_particles, thread_per_block>>>
                        (dev_p, dev_grid, dev_para, dev_rigid_bodies);
            }

            if (block_grid)
            {
                grid_update_kernel <<<block_grid, thread_per_block>>>
                        (dev_grid, dev_para);
            }

            if (block_particles)
            {
                grid_to_particle_kernel <<<block_particles, thread_per_block>>>
                        (dev_p, dev_grid, dev_para, dev_rigid_bodies);
            }

            if (pa.n_rigid_bodies)
            {
                rigid_pre_advection_kernel <<<1, pa.n_rigid_bodies>>>
                        (dev_para, dev_rigid_bodies);
            }

            if (block_rigid_particles)
            {
                rigid_advection_kernel <<<block_rigid_particles, thread_per_block>>>
                        (dev_para, dev_rigid_bodies,
                                dev_rigid_particles);
            }

            cuda_check_error();
        }
    }

    Solver::~Solver()
    {
        cudaFree(dev_p);
        cudaFree(dev_grid);
        cudaFree(dev_para);
        cudaFree(dev_rigid_bodies);
        cudaFree(dev_rigid_particles);
    }

    const Particle* Solver::particles_info()
    {
        cudaMemcpy(particles.get(), dev_p,
                pa.n_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
        cuda_check_error();
        return particles.get();
    }

    [[maybe_unused]] const Rigid_body* Solver::rigid_info()
    {
        cudaMemcpy(rigid_bodies, dev_rigid_bodies,
                pa.n_rigid_bodies * sizeof(Rigid_body), cudaMemcpyDeviceToHost);
        cuda_check_error();
        return rigid_bodies;
    }

    void Solver::update_cuda_memory()
    {
        cudaMemcpy(dev_p, particles.get(),
                pa.n_particles * sizeof(Particle), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rigid_particles, rigid_particles.get(),
                pa.n_rigid_particles * sizeof(Rigid_Particle),
                cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rigid_bodies, rigid_bodies,
                pa.n_rigid_bodies * sizeof(Rigid_body),
                cudaMemcpyHostToDevice);

        Parameter para;
        para.n_particles = pa.n_particles;
        para.n_grid = pa.n_grid;
        para.dt = pa.dt;
        para.n_rigid_particles = pa.n_rigid_particles;
        para.n_rigid_bodies = pa.n_rigid_bodies;
        para.dx = pa.dx;
        para.inv_dx = pa.inv_dx;
        para.p_vol = pa.p_vol;
        para.p_rho = pa.p_rho;
        para.p_mass = pa.p_mass;
        para.gravity = pa.gravity;
        cudaMemcpy(dev_para, &para, sizeof(Parameter),
                cudaMemcpyHostToDevice);
        cuda_check_error();
    }

    void
    Solver::add_particle(Real x, Real y, Type type, Color c, Real h, Real vx,
            Real vy)
    {
        if (pa.n_particles >= m_particles) return;
        auto& p = particles[pa.n_particles];
        p = Particle{};
        p.x[0] = x;
        p.x[1] = y;
        p.type = type;
        p.color = c;
        p.v[0] = vx;
        p.v[1] = vy;
        p.h = h;
        pa.n_particles++;
    }

    void Solver::add_rigid_particle(Real x, Real y, Real ux, Real uy, int id)
    {
        if (pa.n_rigid_particles >= m_rigid_particles) return;
        auto& pr = rigid_particles[pa.n_rigid_particles];
        pr = Rigid_Particle{};
        pr.id = id;
        pr.x = Vec(x, y);
        pr.uaxis = Vec(ux, uy);
        pa.n_rigid_particles++;
    }

    void
    Solver::add_line(Real x1, Real y1, Real x2, Real y2, Real d, SurfaceType b,
            Real mu, int id)
    {
        if (pa.n_rigid_bodies > 15) return;
        Vec line_vec((x2 - x1), (y2 - y1));
        Vec uaxis(-(y2 - y1), (x2 - x1));
        auto length = line_vec.norm();
        line_vec = line_vec * Real(1.0 / length);
        uaxis = uaxis * Real(1.0 / uaxis.norm());
        auto segment_num = static_cast<int>(length / pa.dx);
        auto bid = id;
        if (bid == -1) bid = pa.n_rigid_bodies;
        for (auto i = 0; i <= segment_num / 2; i++)
        {
            add_rigid_particle(
                    x1 + utils::narrow_cast<Real>(i) * pa.dx * line_vec[0],
                    y1 + utils::narrow_cast<Real>(i) * pa.dx * line_vec[1],
                    uaxis[0], uaxis[1], bid);
        }
        for (auto i = segment_num / 2 + 1; i <= segment_num; i++)
        {
            add_rigid_particle(
                    x1 + utils::narrow_cast<Real>(i) * pa.dx * line_vec[0],
                    y1 + utils::narrow_cast<Real>(i) * pa.dx * line_vec[1],
                    uaxis[0], uaxis[1], bid);
        }
        if (id != -1) return;
        auto& rb = rigid_bodies[pa.n_rigid_bodies];
        rb.x = Vec((x1 + x2) / 2.0f, (y1 + y2) / 2.0f);
        rb.v = Vec(0);
        rb.w = 0;
        rb.theta = 0;
        rb.density = d;
        rb.b = b;
        rb.mu = mu;
        rb.bt = BodyType::line;
        pa.n_rigid_bodies++;
    }

    void Solver::add_circle(Real centerX, Real centerY, Real r, SurfaceType b,
            Real mu)
    {
        auto& rb = rigid_bodies[pa.n_rigid_bodies];
        rb.x = Vec(centerX, centerY);
        rb.r = r;
        rb.v = Vec(0);
        rb.w = 0;
        rb.theta = 0;
        rb.density = 1;
        rb.b = b;
        rb.mu = mu;
        rb.bt = BodyType::circle;
        pa.n_rigid_bodies++;
    }

    [[maybe_unused]] void
    Solver::add_waterwheel(float centerX, float centerY, float R1, float R2,
            float R3, float d_theta, int n, Real d, SurfaceType b, Real mu)
    {
        static const float pi = 3.1415926;
        for (auto i = 0; i < n; i++)
        {
            auto theta = float(2.0f * pi * utils::narrow_cast<Real>(i) /
                               utils::narrow_cast<Real>(n));
            auto x1 = centerX + R1 * cos(theta);
            auto y1 = centerY + R1 * sin(theta);
            auto x2 = centerX + R3 * cos(theta - d_theta);
            auto y2 = centerY + R3 * sin(theta - d_theta);
            add_line(x1, y1, x2, y2, 1e-5, b, mu, pa.n_rigid_bodies);
        }
        auto& rb = rigid_bodies[pa.n_rigid_bodies];
        rb.x = Vec(centerX, centerY);
        rb.v = Vec(0);
        rb.w = 0;
        rb.theta = 0;
        rb.density = d;
        rb.b = b;
        rb.mu = mu;
        rb.bt = BodyType::line;
        pa.n_rigid_bodies++;
        add_circle(centerX, centerY, R2);
    }

    void Solver::add_cube(Real x, Real y, Real cw, Real ch, int n, Type type,
            Color c, Real h, Real vx, Real vy)
    {
        for (auto i = 0; i < n; i++)
            add_particle(x + cw * Real(utils::rand_real()),
                    y + ch * Real(utils::rand_real()), type, c, h, vx, vy);
    }

    __device__ Vec
    velocity_sub_projection(const Vec& v, const Vec& n, SurfaceType b, Real mu)
    {
        if (v.norm() <= 0.0) return Vec(0);
        if (b == SurfaceType::sticky) return Vec(0);
        auto vt = v - n * v.dot(n);
        if (b == SurfaceType::slip) return vt;
        if (v.dot(n) <= 0)
        {
            if (vt.norm() <= 0.0) return Vec(0);
            auto zeta = max(0.0f, vt.norm() + mu * v.dot(n));
            return vt * Real(1.0 / vt.norm() * zeta);
        }
        else return v;
    }

    __device__ Vec
    velocity_projection(const Rigid_body& rb, const Particle& p, const Vec& x)
    {
        auto r_pos = x - rb.x;
        Vec v_w(-r_pos[1], r_pos[0]);
        if (r_pos.norm() <= 0.0) v_w = Vec(0);
        else v_w = v_w * Real(1.0 / v_w.norm());
        auto Vr = rb.v + v_w * rb.w * r_pos.norm();
        Vec normal;
        if (rb.bt == BodyType::circle) // r_pos.norm() > 0
            normal = r_pos * Real(1.0 / r_pos.norm());
        else normal = p.normal;
        return (Vr + velocity_sub_projection(p.v - Vr, normal, rb.b, rb.mu));
    }

    __device__ void
    apply_impulse(Rigid_body& rb, const Vec& impulse, const Vec& grid_pos)
    {
        if (rb.bt == BodyType::circle) return;
        auto delta_w = (grid_pos - rb.x).cross(impulse) / rb.density;

#if __CUDA_ARCH__ >= 600
        atomicAdd(&rb.w, delta_w);
#endif

    }

    __device__ Mat get_rotation_matrix(Vec v)
    {
        auto v_norm = v.norm();
        auto cos_phi = v[1] / v_norm;
        auto sin_phi = v[0] / v_norm;
        return {
                cos_phi, -sin_phi,
                sin_phi, cos_phi
        };
    }
}
