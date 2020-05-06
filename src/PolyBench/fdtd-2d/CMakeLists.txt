add_hipcl_test(fdtd2d fdtd2d Passed fdtd2d.cu)
target_include_directories(fdtd2d PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

