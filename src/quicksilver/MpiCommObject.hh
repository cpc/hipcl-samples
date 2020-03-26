#ifndef MPI_COMM_OBJECT_HH
#define MPI_COMM_OBJECT_HH

#include "CommObject.hh"

#include "utilsMpi.hh"
#include <set>
#include <vector>

#include "DecompositionObject.hh"
#include "Long64.hh"
#include "MeshPartition.hh"

class MpiCommObject : public CommObject {
public:
  MpiCommObject(const MPI_Comm &comm, const DecompositionObject &ddc);

  void exchange(MeshPartition::MapType &cellInfo,
                const std::vector<int> &nbrDomain,
                std::vector<std::set<Long64>> sendSet,
                std::vector<std::set<Long64>> recvSet);

  void exchange(std::vector<FacetPair> sendBuf,
                std::vector<FacetPair> &recvBuf);

private:
  MPI_Comm _comm;
  DecompositionObject _ddc;
};

#endif
