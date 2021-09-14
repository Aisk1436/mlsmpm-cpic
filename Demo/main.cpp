#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "MPM_Solver.cuh"
#include "utils.h"

const int window_x = 512;
const int window_y = 512;
const char window_name[] = "mpm";

int quality = 1;
float hardness = 0.3;
int n_particles = 9000 * quality * quality;
float dt = 1e-4f / float(quality);
int n_grid = 128 * quality;
bool cpic_flag = true;

std::unique_ptr<Mpm::Solver> solver;

void init_GL(GLFWwindow**);

void init();

void main_loop(GLFWwindow*);

void render(GLFWwindow*);

void
key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode,
        int action, [[maybe_unused]] int mods);

void draw_dot(GLdouble, GLdouble, GLdouble, GLdouble, GLdouble, GLdouble = 0.0);

void error_callback(int, const char*);

int main()
try
{
    solver = std::make_unique<Mpm::Solver>(n_particles, n_grid, dt);
    GLFWwindow* window;
    init_GL(&window);
    init();
    main_loop(window);
    glfwTerminate();
    return 0;
}
catch (std::exception& e)
{
    std::cerr << e.what() << "\n";
    glfwTerminate();
    utils::keep_window_open();
    exit(1);
}

void init_GL(GLFWwindow** window)
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) utils::error("glfwInit() failure");
    glfwWindowHint(GLFW_SAMPLES, 8);
    (*window) = glfwCreateWindow(window_x, window_y, window_name, nullptr,
            nullptr);
    glfwSetKeyCallback(*window, key_callback);
    glfwMakeContextCurrent(*window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) utils::error("glewInit() failure");
}

void init()
{
    glClearColor(0.25, 0.25, 0.25, 1.0);
    solver->reset(n_particles, n_grid, dt);
    auto n = n_particles / 3;

    solver->add_cube(0.3, 0.05, 0.2, 0.2, n,
            Mpm::Type::liquid, Mpm::Color{ 0.616, 0.8, 0.878 });
    solver->add_cube(0.4, 0.37, 0.2, 0.2, n,
            Mpm::Type::elastic, Mpm::Color{ 0.929, 0.333, 0.231 }, hardness);
    solver->add_cube(0.5, 0.69, 0.2, 0.2, n,
            Mpm::Type::snow, Mpm::Color{ 0.9, 0.9, 0.9 });

    if (cpic_flag)
    {
        solver->add_line(0.975, 0.5, 0.025, 0.5, 1e-4,
                Mpm::SurfaceType::separate);
        solver->add_line(0.5, 0.975, 0.5, 0.025, 1e-4,
                Mpm::SurfaceType::separate);
    }

    solver->update_cuda_memory();
}

void main_loop(GLFWwindow* window)
{
    while (!glfwWindowShouldClose(window))
    {
        auto frame_interval = int(2e-3 / (1e-4 / quality));
        solver->advance(frame_interval);
        render(window);
        glfwPollEvents();
    }
}

void render(GLFWwindow* window)
{
    constexpr float pi = 3.1415926;
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_MULTISAMPLE);
    auto ps = solver->particles_info();
    double dot_size;
    switch (quality)
    {
    case 1:
        dot_size = 4.0;
        break;
    case 2:
        dot_size = 3.0;
        break;
    case 3:
        dot_size = 2.0;
        break;
    case 4:
        dot_size = 1.5;
        break;
    default:
        dot_size = 1.0;
    }

    for (auto i = 0; i < n_particles; i++)
    {
        auto& p = ps[i];
        draw_dot(p.x[0] * 2 - 1, p.x[1] * 2 - 1,
                p.color.r, p.color.g, p.color.b, dot_size);
    }

    if (cpic_flag)
    {
        const Mpm::Rigid_body* rbs = solver->rigid_info();
        float rigid_theta = rbs->theta;

        glColor3d(1.0, 1.0, 1.0);
        glBegin(GL_LINES);
        glVertex2d(-0.95 * cos(rigid_theta), -0.95 * sin(rigid_theta));
        glVertex2d(0.95 * cos(rigid_theta), 0.95 * sin(rigid_theta));
        rigid_theta += pi / 2;
        glVertex2d(-0.95 * cos(rigid_theta), -0.95 * sin(rigid_theta));
        glVertex2d(0.95 * cos(rigid_theta), 0.95 * sin(rigid_theta));
        glEnd();
    }

    glfwSwapBuffers(window);
}

static void
key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode,
        int action, [[maybe_unused]] int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        init();
    if (key == GLFW_KEY_SLASH && action == GLFW_PRESS)
    {
        cpic_flag = !cpic_flag;
        init();
    }
    if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS)
    {
        if (quality < 5)
        {
            quality++;
            n_particles = 9000 * quality * quality;
            dt = 1e-4f / utils::narrow_cast<float>(quality);
            n_grid = 128 * quality;
        }
        init();
    }
    if (key == GLFW_KEY_COMMA && action == GLFW_PRESS)
    {
        if (quality > 1)
        {
            quality--;
            n_particles = 9000 * quality * quality;
            dt = 1e-4f / utils::narrow_cast<float>(quality);
            n_grid = 128 * quality;
        }
        init();
    }
}

void draw_dot(GLdouble x, GLdouble y,
        GLdouble r, GLdouble g, GLdouble b, GLdouble size)
{
    constexpr auto n_side = 8;
    static GLdouble pi = 3.1415926;
    static GLdouble sin_tab[n_side];
    static GLdouble cos_tab[n_side];

    static auto init_flag = true;
    if (init_flag)
    {
        for (auto i = 0; i < n_side; i++)
        {
            sin_tab[i] = sin(i * pi * 2 / n_side);
            cos_tab[i] = cos(i * pi * 2 / n_side);
        }
        init_flag = false;
        (void)init_flag;
    }

    glColor3d(r, g, b);
    if (size == 0.0)
    {
        glBegin(GL_POINTS);
        glVertex2d(x, y);
        glEnd();
        return;
    }
    auto radius_x = size / window_x / 2;
    auto radius_y = size / window_y / 2;
    glBegin(GL_POLYGON);
    for (auto i = 0; i < n_side; i++)
    {
        glVertex2d(x + radius_x * cos_tab[i],
                y + radius_y * sin_tab[i]);
    }
    glEnd();
}

void error_callback(int err, const char* description)
{
    utils::error(description, err);
}