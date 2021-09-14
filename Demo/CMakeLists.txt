add_executable(Demo main.cpp)

find_package(GLEW REQUIRED)
target_link_libraries(Demo LINK_PUBLIC GLEW::GLEW)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(Demo LINK_PUBLIC glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(Demo LINK_PUBLIC OpenGL::GL)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(Demo PRIVATE fmt::fmt-header-only)

target_include_directories(Demo
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../Matrix)

target_link_libraries (Demo LINK_PUBLIC Solver)