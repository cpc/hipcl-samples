add_hipcl_test(doitgen doitgen Passed doitgen.cu)
target_include_directories(doitgen PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

