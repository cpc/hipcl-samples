add_hipcl_test(covariance covariance Passed covariance.cu)
target_include_directories(covariance PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utilities>)

