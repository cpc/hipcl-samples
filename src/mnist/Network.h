#ifndef Network_H
#define Network_H

#include "Node.h"

typedef vector<Node> Level;

class Network {

public:
  // default constructor
  Network();
  // constructor that takes in a certain topology and then builds the Network
  Network(const vector<unsigned> &topol);
  // constructor that builds a Network from a file
  Network(string filename);
  // initializes a Network with a certain topology
  void init(const vector<unsigned> &topol);
  // allocates GPU memory
  void allocmemGPU();
  // frees memory on gpu
  void deallocmemGPU();
  // copies memory from the gpu to the cpu
  void copyGpuToCpu();
  // feeds a vector of inputs through the Network
  void FdFwd(vector<double> &inVals);
  // backpropogates on expected target values
  void backProp(const vector<double> &goalVals);
  // gets the values at the output nuerons
  void getResults(vector<double> &resultVals) const;
  // outputs the Network to a file
  int outputToFile(string filename);

  // parrallel feed forward
  void FdFwdParallel(double *invals);
  // parrallel Back Propogation
  void backPropParallel(double *goalVals);
  // gets the reulst from the GPU and stores it into a
  void getResultsFromGPU();

  // contains the Network
  vector<Level> m_levels; // m_levels[levelNum][NodeNum]
  double *results_h;

private:
  double n_error;
  double m_AverageError;
  double m_AverageSmoothingFactor;
  int levels;

  // device variables to store the Network on the GPU
  int *topol_d;
  double *weights_d;
  double *derivWeights_d;
  double *outputval_d;
  double *gradients_d;
  double *results_d;
  int *error_d;
  double *goalVals_d;
};

#endif
