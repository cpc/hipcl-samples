add_hipcl_test(gramschmidt gramschmidt Passed gramschmidt.cu)
target_include_directories(gramschmidt PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

