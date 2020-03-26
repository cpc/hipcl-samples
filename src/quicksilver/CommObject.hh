#ifndef COMM_OBJECT_HH
#define COMM_OBJECT_HH

#include <set>

#include "FacetPair.hh"
#include "Long64.hh"
#include "MeshPartition.hh"
#include <vector>

class CommObject {
public:
  virtual ~CommObject(){};
  virtual void exchange(MeshPartition::MapType &cellInfo,
                        const std::vector<int> &nbrDomain,
                        std::vector<std::set<Long64>> sendSet,
                        std::vector<std::set<Long64>> recvSet) = 0;
  virtual void exchange(std::vector<FacetPair> sendBuf,
                        std::vector<FacetPair> &recvBuf) = 0;
};

#endif
