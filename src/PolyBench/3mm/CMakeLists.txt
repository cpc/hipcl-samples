add_hipcl_test(3mm 3mm Passed 3mm.cu)
target_include_directories(3mm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

