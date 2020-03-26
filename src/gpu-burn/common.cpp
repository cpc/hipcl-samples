/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#include "common.h"
#include <iostream>

// ---------------------------------------------------------------------------
namespace gpuburn {

int checkError(hipError_t err, std::string desc) {
  if (err == hipSuccess)
    return 0;

  std::string errStr = hipGetErrorString(err);
  std::string errorMessage = "";
  if (desc == "")
    // throw "Error: " + errStr + "\n";
    std::cerr << "Error: " << errStr << "\n";
  else
    std::cerr << "Error in " << desc << ": " << errStr << "\n";
  // throw "Error in \"" + desc + "\": " + errStr + "\n";

  return err;
}

}; // namespace gpuburn

// ---------------------------------------------------------------------------
