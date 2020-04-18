add_hipcl_test(jacobi2D jacobi2D Passed jacobi2D.cu)
target_include_directories(jacobi2D PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

