foreach(UBYTE t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte  train-labels-idx1-ubyte)
  message(STATUS "downloading ${UBYTE}")
  file(DOWNLOAD "https://github.com/ethancaballero/cplusplus-cuda-mnist-neural-network/raw/master/${UBYTE}" "${CMAKE_CURRENT_BINARY_DIR}/${UBYTE}")
endforeach()

add_hipcl_binary(mnist-nn Node.cpp Network.cpp main.cpp)
