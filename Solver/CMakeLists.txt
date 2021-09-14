add_library(Solver STATIC MPM_Solver.cu MPM_Solver.cuh)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(Solver PRIVATE fmt::fmt-header-only)

target_include_directories(Solver
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(Solver
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../Matrix)

set_target_properties(Solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60)

target_compile_options(Solver PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)