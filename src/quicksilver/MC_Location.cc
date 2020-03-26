#include "MC_Location.hh"
#include "DeclareMacro.hh"
#include "MC_Domain.hh"
#include "MonteCarlo.hh"

//  Return a reference to the domain for this location.

HOST_DEVICE
const MC_Domain &MC_Location::get_domain(MonteCarlo *mcco) const {
  return mcco->domain[domain];
}

HOST_DEVICE_END
