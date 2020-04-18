add_hipcl_test(gemver gemver Passed gemver.cu)
target_include_directories(gemver PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

