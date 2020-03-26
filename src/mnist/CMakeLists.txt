add_hipcl_binary(mnist-nn Node.cpp Network.cpp main.cpp)
#target_compile_options(mnist-nn PRIVATE -O3 -march=native)
