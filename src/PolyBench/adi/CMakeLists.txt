add_hipcl_test(adi adi Passed adi.cu)
target_include_directories(adi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

