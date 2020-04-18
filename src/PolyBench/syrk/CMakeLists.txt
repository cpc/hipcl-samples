add_hipcl_test(syrk syrk Passed syrk.cu)
target_include_directories(syrk PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

