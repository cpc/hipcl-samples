add_hipcl_test(atax atax Passed atax.cu)
target_include_directories(atax PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

