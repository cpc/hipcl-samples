## hipcl-samples

This repository contains 3rd party examples ported to HIPCL.

For license and copyright, please check each respective example's source.

The src/CMakeLists.txt and files outside src/ subdirectory are licensed under MIT, see LICENSE
#### Requirements

* compiled HIPCL
* CMake 3.4+
* Clang with HIPCL's patches

#### Compiling

configure with:

    cmake -DCMAKE_CXX_COMPILER=/opt/hipcl/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/hipcl/llvm/bin/clang /path/to/hipcl-samples

if you installed HIPCL into non-default location HPREFIX, use:

    cmake -DHIPCL_PREFIX=$HPREFIX -DCMAKE_CXX_COMPILER=$HPREFIX/llvm/bin/clang++ -DCMAKE_C_COMPILER=$HPREFIX/llvm/bin/clang /path/to/hipcl-samples
