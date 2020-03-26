#ifndef Node_H
#define Node_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define e 2.71828182845904523536

using namespace std;

class Node;
typedef vector<Node> Level;

struct Link {
  double weight;
  double derivWeight;
};

class Node {

public:
  Node(unsigned OutCount, unsigned idx);
  Node(double outputVal, vector<Link> outputWeights, unsigned idx,
       double gradient);
  void setOut(double val) { m_outputVal = val; }
  double getOutVal() const { return m_outputVal; }
  void FdFwd(const Level &prevLevel);
  void calcOutGradients(double goalVal);
  void calcHiddenGradients(const Level &nextLevel);
  void updateInWeights(Level &prevLevel);
  void outputWeightsToFile(string filename);
  double m_outputVal;
  vector<Link> m_outputWeights;
  unsigned m_idx;
  double m_gradient;

private:
  static double transFunc(double x);
  static double transFuncDeriv(double x);
  static double transFunc2(double x);
  static double transFuncDeriv2(double x);
  static double randomWeight(void) {
    double ran = rand() / double(RAND_MAX);
    ran *= 2.0;
    ran -= 1.0;
    return ran * .11;
  }
  double sumDerivWeights(const Level &nextLevel) const;

  static double eta;
  static double alpha;
};

#endif
