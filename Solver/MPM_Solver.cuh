#ifndef MPM_SOLVER_LIBRARY_CUH
#define MPM_SOLVER_LIBRARY_CUH

#include "Matrix.h"

namespace Mpm
{
    using Real = float;
    using Vec = Vec2d<Real>;
    using Mat = Mat2d<Real>;
    using Veci = Vec2d<int>;
    using Vec3d = Vec3d<Real>;
    using Mat3d = Mat3d<Real>;

    enum class Type
    {
        liquid, elastic, snow
    };

    enum class SurfaceType
    {
        sticky, slip, separate
    };

    enum class BodyType
    {
        line, circle
    };

    struct Color
    {
        Real r, g, b;
    };

    struct Particle
    {
        Vec x;
        Vec v;
        Type type;
        Color color;
        Real h;
        Mat C;
        Mat F{ 1.0 };
        Mat affine;
        Real Jp{ 1.0 };
        Real J{ 1.0 };

        int A_states;
        int T_states;
        Real d;
        Vec normal;
    };

    struct Rigid_body
    {
        Vec x;
        Vec v;
        Real w;
        Real theta;

        SurfaceType b;
        Real mu;
        Real density;

        BodyType bt;
        Real r;
    };

    struct Rigid_Particle
    {
        int id{};
        Vec x;
        Vec uaxis;
    };

    struct Grid
    {
        Vec v;
        Real m{ 0.0 };

        Real d{ 0.0 };
        int closest_rigid_body{ -1 };       // rigid body id
        int A_states{ 0 };
        int T_states{ 0 };

        int mutex;

        CUDA_CALLABLE_MEMBER void init()
        {
            v = Vec();
            m = 0.0;
            d = 0.0;
            closest_rigid_body = -1;
            A_states = 0;
            T_states = 0;
            mutex = 0;
        }
    };

    struct Parameter
    {
        int n_particles;
        int n_rigid_particles;
        int n_rigid_bodies;
        int n_grid;
        Real dt;
        Real dx;
        Real inv_dx;
        Vec gravity;
        Real p_vol;
        Real p_rho;
        Real p_mass;
    };

    class Solver
    {
        int m_particles{};
        int m_rigid_particles{};
        Parameter pa{};
        std::unique_ptr<Particle[]> particles{};
        std::unique_ptr<Rigid_Particle[]> rigid_particles{};
        Rigid_body rigid_bodies[32];
        Particle* dev_p{};
        Grid* dev_grid{};
        Parameter* dev_para{};
        Rigid_Particle* dev_rigid_particles{};
        Rigid_body* dev_rigid_bodies{};

    public:
        Solver(int, int, Real);

        ~Solver();

        void init();

        void reset(int, int, Real);

        void add_particle(Real x, Real y, Type type,
                Color c, Real h = 0.3, Real vx = 0, Real vy = 0);

        void add_rigid_particle(Real x, Real y, Real ux, Real uy, int id);

        void add_line(Real x1, Real y1, Real x2, Real y2, Real d,
                SurfaceType b = SurfaceType::separate, Real mu = 0.0,
                int id = -1);

        void add_circle(Real centerX, Real centerY, Real r,
                SurfaceType b = SurfaceType::separate, Real mu = 0.0);

        [[maybe_unused]] void
        add_waterwheel(float centerX, float centerY, float R1, float R2,
                float R3, float d_theta, int n,
                Real d, SurfaceType b = SurfaceType::separate, Real mu = 0.0);

        void add_cube(Real x, Real y, Real cw, Real ch, int n,
                Type type, Color c, Real h = 0.3, Real vx = 0, Real vy = 0);

        void advance(int);

        [[maybe_unused]] [[nodiscard]] int particles_number() const
        {
            return pa.n_particles;
        }

        [[maybe_unused]] [[nodiscard]] int bodies_number() const
        {
            return pa.n_rigid_bodies;
        }

        const Particle* particles_info();

        [[maybe_unused]] const Rigid_body* rigid_info();

        void update_cuda_memory();
    };
}

#endif //MPM_SOLVER_LIBRARY_CUH
