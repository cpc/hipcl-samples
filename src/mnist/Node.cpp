#include "Node.h"

double Node::eta = 0.39;  // learning speed
double Node::alpha = 0.1; // momentum, multiplier of last derivWeight

/*New constructor for use with the loader*/
Node::Node(double outputVal, vector<Link> outputWeights, unsigned idx,
           double gradient) {
  m_outputVal = outputVal;
  vector<Link>::iterator it;
  for (it = outputWeights.begin(); it != outputWeights.end(); it++) {
    m_outputWeights.push_back(*it);
  }
  m_idx = idx;
  m_gradient = gradient;
}

void Node::updateInWeights(Level &prevLevel) {
  // weights in the Link container in the Nodes in preceding level need to be
  // updated

  for (unsigned n = 0; n < prevLevel.size(); ++n) {

    Node &Node = prevLevel[n];
    double oldderivWeight = Node.m_outputWeights[m_idx].derivWeight;

    double newderivWeight =
        // individual input, magnified by the gradient and train rate
        eta * Node.getOutVal() * m_gradient + alpha * oldderivWeight;

    Node.m_outputWeights[m_idx].derivWeight = newderivWeight;
    Node.m_outputWeights[m_idx].weight += newderivWeight;
  }
}

double Node::sumDerivWeights(const Level &nextLevel) const {
  double sum = 0.0;

  for (unsigned n = 0; n < nextLevel.size() - 1; ++n) {
    sum += m_outputWeights[n].weight * nextLevel[n].m_gradient;
  }
  return sum;
}

void Node::calcOutGradients(double goalVal) {
  double delta = goalVal - m_outputVal;
  m_gradient = delta * Node::transFuncDeriv(m_outputVal);
}

void Node::calcHiddenGradients(const Level &nextLevel) {
  double dow = sumDerivWeights(nextLevel);
  m_gradient = dow * Node::transFuncDeriv(m_outputVal);
  m_gradient /= nextLevel.size();
}

double Node::transFunc(double y) {
  // returns between [-1 .. 1]
  return tanh(y);
}

double Node::transFunc2(double y) {
  // returns between [-1 .. 1]
  return (1 / (1 + pow(e, -y)));
}

double Node::transFuncDeriv(double x) {
  // deriv tanh
  return 1.0 - (x * x);
}

double Node::transFuncDeriv2(double x) {
  // deriv tanh
  return (1 - transFunc2(x)) * transFunc2(x);
}

Node::Node(unsigned OutCount, unsigned idx) {
  for (unsigned c = 0; c < OutCount; ++c) {
    m_outputWeights.push_back(Link());
    m_outputWeights.back().weight = randomWeight();
  }
  m_idx = idx;
}

void Node::FdFwd(const Level &prevLevel) {

  double sum = 0.0;
  // loop through all the previous levels outputs (which are now inputs)

  for (unsigned n = 0; n < prevLevel.size(); ++n) {
    sum +=
        prevLevel[n].getOutVal() * prevLevel[n].m_outputWeights[m_idx].weight;
  }

  sum /= (prevLevel.size() / 2.0);
  m_outputVal = Node::transFunc(sum);
}
