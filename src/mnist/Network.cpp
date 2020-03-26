#include "Network.h"
#include "hip/hip_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

Network::Network() {}

Network::Network(const vector<unsigned> &topol) { init(topol); }

void Network::init(const vector<unsigned> &topol) {

  results_h = new double[10];

  m_levels.clear();

  unsigned numLevels = topol.size();
  levels = topol.size();

  for (unsigned levelNum = 0; levelNum < numLevels; ++levelNum) {
    m_levels.push_back(Level());

    // fill the level with nuerons
    // loop since each level has a bias nueron

    unsigned OutCount = levelNum == topol.size() - 1 ? 0 : topol[levelNum + 1];

    for (unsigned NodeNum = 0; NodeNum <= topol[levelNum]; ++NodeNum) {
      // make a new Nueron
      m_levels.back().push_back(Node(OutCount, NodeNum));
    }
    // force the bias nodes's output value to 1.0
    m_levels.back().back().setOut(1.0);
  }
}

void Network::allocmemGPU() {

  hipMalloc((void **)&topol_d, sizeof(int) * levels);
  int topol_h[levels];

  int osize = 0;
  int wsize = 0;
  for (int i = 0; i < levels; i++) {
    topol_h[i] = m_levels[i].size();
    osize += m_levels[i].size();
  }
  hipMemcpy(topol_d, &topol_h, sizeof(int) * levels, hipMemcpyHostToDevice);

  for (int l = 0; l < levels; l++) {
    for (int n = 0; n < topol_h[l]; n++) {
      wsize += m_levels[l][n].m_outputWeights.size();
    }
  }

  double *weights_h = new double[wsize];
  double *derivWeights_h = new double[wsize];
  double *outputval_h = new double[osize];

  int wcounter = 0;
  int lcounter = 0;

  for (int l = 0; l < levels; l++) {
    for (int n = 0; n < topol_h[l]; n++) {
      for (int i = 0; i < m_levels[l][n].m_outputWeights.size(); i++) {

        weights_h[i + wcounter] = m_levels[l][n].m_outputWeights[i].weight;
        derivWeights_h[i + wcounter] =
            m_levels[l][n].m_outputWeights[i].derivWeight;
      }
      wcounter += m_levels[l][n].m_outputWeights.size();
      outputval_h[lcounter + n] = m_levels[l][n].m_outputVal;
    }
    lcounter += topol_h[l];
  }

  hipMalloc((void **)&goalVals_d, sizeof(double) * 10);
  hipMalloc((void **)&weights_d, sizeof(double) * wsize);
  hipMalloc((void **)&derivWeights_d, sizeof(double) * wsize);
  hipMalloc((void **)&outputval_d, sizeof(double) * osize);
  hipMalloc((void **)&gradients_d, sizeof(double) * osize);
  hipMalloc((void **)&error_d, sizeof(int));
  hipDeviceSynchronize();

  hipMemcpy(weights_d, weights_h, sizeof(double) * wsize,
            hipMemcpyHostToDevice);
  hipMemcpy(derivWeights_d, derivWeights_h, sizeof(double) * wsize,
            hipMemcpyHostToDevice);
  hipMemcpy(outputval_d, outputval_h, sizeof(double) * osize,
            hipMemcpyHostToDevice);
  hipDeviceSynchronize();

  delete[] weights_h;
  delete[] derivWeights_h;
  delete[] outputval_h;
}

void Network::deallocmemGPU() {
  hipDeviceSynchronize();
  hipFree(weights_d);
  hipFree(derivWeights_d);
  hipFree(topol_d);
  hipFree(outputval_d);
  hipFree(gradients_d);
  hipFree(error_d);
  hipFree(goalVals_d);
}

void Network::copyGpuToCpu() {

  int topol_h[levels];

  int osize = 0;

  for (int i = 0; i < levels; i++) {

    osize += m_levels[i].size();
  }

  hipMemcpy(topol_h, topol_d, sizeof(int) * levels, hipMemcpyDeviceToHost);
  vector<unsigned> topol;
  for (int i = 0; i < levels; i++) {

    topol_h[i]--;
    topol.push_back(topol_h[i]);
    topol_h[i]++;
  }

  init(topol);

  int wsize = 0;

  for (int l = 0; l < levels; l++) {

    for (int n = 0; n < topol_h[l]; n++) {

      wsize += m_levels[l][n].m_outputWeights.size();
    }
  }

  double *weights_h = new double[wsize];
  double *derivWeights_h = new double[wsize];
  double *outputval_h = new double[osize];

  int wcounter = 0;
  int lcounter = 0;

  hipMemcpy(weights_h, weights_d, sizeof(double) * wsize,
            hipMemcpyDeviceToHost);
  hipMemcpy(derivWeights_h, derivWeights_d, sizeof(double) * wsize,
            hipMemcpyDeviceToHost);
  hipMemcpy(outputval_h, outputval_d, sizeof(double) * osize,
            hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

  for (int l = 0; l < levels; l++) {
    for (int n = 0; n < topol_h[l]; n++) {
      for (int i = 0; i < m_levels[l][n].m_outputWeights.size(); i++) {
        m_levels[l][n].m_outputWeights[i].weight = weights_h[i + wcounter];
        m_levels[l][n].m_outputWeights[i].derivWeight =
            derivWeights_h[i + wcounter];
      }

      wcounter += m_levels[l][n].m_outputWeights.size();
      m_levels[l][n].m_outputVal = outputval_h[lcounter + n];
    }
    lcounter += topol_h[l];
  }

  delete[] weights_h;
  delete[] derivWeights_h;
  delete[] outputval_h;
}

/*takes a file, uses a vector representation of that file, then creates Nodes.*/
/*file is going to be lengths separated by space, \n\n weights separated by
 * \n\n,- error.*/
/*The "loader."*/
Network::Network(string filename) {
  FILE *fp;
  long fsize;

  char *buf;
  fp = fopen(filename.c_str(), "r");

  /*code to allocate and fill a buffer with the file contents.*/
  fseek(fp, 0, SEEK_END);
  fsize = ftell(fp);
  rewind(fp);
  buf = (char *)malloc(fsize * sizeof(char));
  fread(buf, 1, fsize, fp);
  fclose(fp);
  char *initialbuf = buf;
  unsigned numLevels;
  /*This gets the number of levels based on the layout of the file, then creates
   * an appropriate vector based on that size.*/

  memcpy(&numLevels, buf, sizeof(unsigned));

  buf += sizeof(unsigned);

  char *levelVals =
      buf; /*points to how many elements are in the current level.*/
  for (int i = 0; i < numLevels; i++) {
    buf += sizeof(int); /*skip past all the levels to where the first piece of
                           actual data is.*/
  }

  for (unsigned levelNum = 0; levelNum < numLevels; levelNum++) {
    m_levels.push_back(Level());
    double outputVal;
    int outWeightssize;
    vector<Link> outputWeights;
    unsigned idx;
    double gradient;
    int sum;
    memcpy(&sum, levelVals, sizeof(int));
    int counter = 0;
    while (counter != sum) {
      memcpy(&outputVal, buf, sizeof(double));
      buf += sizeof(double);
      memcpy(&outWeightssize, buf, sizeof(int));
      buf = buf + sizeof(int);
      for (int i = 0; i < outWeightssize; i++) {
        double tmp;
        outputWeights.push_back(Link());
        memcpy(&tmp, buf, sizeof(double));
        outputWeights.back().weight = tmp;
        buf = buf + sizeof(double);
        memcpy(&tmp, buf, sizeof(double));
        outputWeights.back().derivWeight = tmp;
        buf = buf + sizeof(double);
        cout << "Vals:"
             << "outWeightssize:" << outWeightssize << " " << counter << " "
             << outputWeights.back().weight << " "
             << outputWeights.back().derivWeight << endl;
      }
      memcpy(&idx, buf, (sizeof(unsigned)));
      buf = buf + sizeof(unsigned);
      memcpy(&gradient, buf, sizeof(double));
      buf = buf + sizeof(double);
      m_levels.back().push_back(Node(outputVal, outputWeights, idx, gradient));
      outputWeights.clear();

      counter++;
    }
    levelVals += sizeof(int);
  }
  free(initialbuf);
}

/*takes in a filename. will output num outputs - outputs - error onto the
 * file.*/
/*Returns 0 on success, -1 on error.*/
/*The "saver."*/
int Network::outputToFile(string filename) {
  FILE *fp;
  /*Assume valid filename*/
  fp = fopen(filename.c_str(), "w");
  if (!fp)
    return -1;

  vector<Level>::iterator it;
  vector<Node>::iterator iter;

  vector<int> NodeSizes;
  uint32_t sum = 0;
  // Get the size of all the Node vectors.
  for (it = m_levels.begin(); it != m_levels.end(); it++) {
    sum += it->size();
    NodeSizes.push_back(it->size()); // used error-checking later.
  }

  unsigned n_levels = m_levels.size();
  cout << "Num_levels:" << n_levels << endl;
  fwrite(&n_levels, sizeof(unsigned), 1, fp);
  for (vector<int>::iterator i = NodeSizes.begin(); i != NodeSizes.end(); i++) {
    /*Put the size of each Node vector into the file.*/;
    int size = *i;
    printf("size:%d\n", size);
    fwrite(&size, sizeof(int), 1, fp);
  }

  // Iterate through levels
  for (it = m_levels.begin(); it != m_levels.end(); it++) {
    // Iterate through Nodes.
    for (iter = it->begin(); iter != it->end(); iter++) {
      // Put the value of the Nodes in the file.
      fwrite(&(iter->m_outputVal), sizeof(double), 1, fp);

      int vecsize = iter->m_outputWeights.size();

      for (vector<Link>::iterator coni = iter->m_outputWeights.begin();
           coni != iter->m_outputWeights.end(); coni++) {
        // vector contents
        fwrite(&(coni->weight), sizeof(double), 1, fp);
        fwrite(&(coni->derivWeight), sizeof(double), 1, fp);
      }

      fwrite(&(iter->m_idx), sizeof(unsigned), 1, fp);

      fwrite(&(iter->m_gradient), sizeof(double), 1, fp);
    }
  }
  fclose(fp);
  return 0;
}

void Network::getResults(vector<double> &resultVals) const {

  resultVals.clear();

  for (unsigned n = 0; n < m_levels.back().size() - 1; ++n) {
    resultVals.push_back(m_levels.back()[n].getOutVal());
  }
}

void Network::FdFwd(vector<double> &inVals) {

  assert(inVals.size() == m_levels[0].size() - 1);

  // Latch the input vals into the input nuerons

  for (unsigned i = 0; i < inVals.size(); ++i) {
    m_levels[0][i].setOut(inVals[i]);
  }

  // Forward Propagation
  for (unsigned levelNum = 1; levelNum < m_levels.size(); ++levelNum) {
    Level &prevLevel = m_levels[levelNum - 1];
    for (unsigned n = 0; n < m_levels[levelNum].size() - 1; ++n) {
      m_levels[levelNum][n].FdFwd(prevLevel);
    }
  }
}

__global__ void latch(double *inputvals, double *nueronoutputvals) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < 784) {
    nueronoutputvals[i] = inputvals[i];
  }
}

__global__ void FdFwdkernel(double *weights, double *nueronoutputvals,
                            int *topol, int currlevel, int outoffset,
                            int woffset) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;

  if (i < (topol[currlevel + 1] - 1)) {
    for (unsigned n = 0; n < topol[currlevel]; ++n) {
      sum += nueronoutputvals[outoffset + n] *
             weights[woffset + (n * (topol[currlevel + 1] - 1)) + i];
    }
    sum /= (topol[currlevel] / 2.0);
    nueronoutputvals[outoffset + topol[currlevel] + i] = tanhf(sum);
  }
}

void Network::FdFwdParallel(double *invals) {

  double *invals_d;

  hipMalloc((void **)&invals_d, sizeof(double) * 784);

  hipMemcpy(invals_d, invals, sizeof(double) * 784, hipMemcpyHostToDevice);

  hipDeviceSynchronize();

  dim3 dim_block_latch(256, 1, 1);
  dim3 dim_grid_latch(4, 1, 1);

  // run a latch kernel
  hipLaunchKernelGGL(latch, dim3(dim_grid_latch), dim3(dim_block_latch), 0, 0,
                     invals_d, outputval_d);
  hipDeviceSynchronize();

  hipFree(invals_d);

  dim3 dim_block(256, 1, 1);
  dim3 dim_grid(8, 1, 1);

  int osize = 0;
  int wsize = 0;

  for (int i = 0; i < levels - 1; i++) {

    dim3 dim_block(256, 1, 1);
    dim3 dim_grid((int)((m_levels[i + 1].size() / 256) + 1), 1, 1);

    hipLaunchKernelGGL(FdFwdkernel, dim3(dim_grid), dim3(dim_block), 0, 0,
                       weights_d, outputval_d, topol_d, i, osize, wsize);
    hipDeviceSynchronize();
    osize += m_levels[i].size();
    wsize += m_levels[i].size() * (m_levels[i + 1].size() - 1);
  }
}

__global__ void getResultskernel(double *results, int outoffset,
                                 double *outputvals) {

  int tid = threadIdx.x;

  if (tid < 10) {
    results[tid] = outputvals[outoffset + tid];
  }
}

void Network::getResultsFromGPU() {

  // Can be stored so that the this does not need to be computed
  int osize;

  for (int i = 0; i < levels - 1; i++) {

    osize += m_levels[i].size();
  }

  hipMalloc((void **)&results_d, sizeof(double) * 10);

  dim3 dim_block(16, 1, 1);
  dim3 dim_grid(1, 1, 1);

  hipLaunchKernelGGL(getResultskernel, dim3(dim_grid), dim3(dim_block), 0, 0,
                     results_d, osize, outputval_d);
  hipDeviceSynchronize();

  for (int i = 0; i < 10; i++) {
    results_h[i] = 0.0;
  }

  hipMemcpy(results_h, results_d, sizeof(double) * 10, hipMemcpyDeviceToHost);
  hipFree(results_d);
}

__global__ void calcOutGradientskernel(double *goalVals, double *outputvals,
                                       double *gradients, int outoffset) {

  int tid = threadIdx.x;

  if (tid < 10) {
    double delta = goalVals[tid] - outputvals[outoffset + tid];
    gradients[outoffset + tid] =
        delta *
        (1.0 - (outputvals[outoffset + tid] * outputvals[outoffset + tid]));
  }
}

__global__ void calcHiddenGradientskernel(double *weights, double *gradients,
                                          int outoffset, int woffset,
                                          int *topol, int currentlevel,
                                          double *outputvals) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < topol[currentlevel]) {

    double dow = 0.0;

    for (int n = 0; n < topol[currentlevel + 1] - 1; ++n) {
      dow += weights[woffset + (i * (topol[currentlevel + 1] - 1)) + n] *
             gradients[outoffset + topol[currentlevel] + n];
    }

    gradients[outoffset + i] =
        dow * (1.0 - (outputvals[outoffset + i] * outputvals[outoffset + i]));
    gradients[outoffset + i] /= topol[currentlevel + 1];
  }
}

__global__ void updateInWeightskernel(double *weights, double *gradients,
                                      double *outputvals, int woffset,
                                      int outoffset, double *derivWeights,
                                      int *topol, int currlevel) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < topol[currlevel] - 1) {

    for (int n = 0; n < topol[currlevel - 1]; ++n) {

      double newderivWeight =
          // individual input , magnified by the gradient and train rate
          .39 * outputvals[outoffset - topol[currlevel - 1] + n] *
              gradients[outoffset + i] +
          .1 * derivWeights[woffset + (n * (topol[currlevel] - 1)) + i];

      derivWeights[woffset + (n * (topol[currlevel] - 1)) + i] = newderivWeight;
      weights[woffset + (n * (topol[currlevel] - 1)) + i] += newderivWeight;
    }
  }
}

void Network::backPropParallel(double *goalVals) {

  hipMemcpy(goalVals_d, goalVals, sizeof(double) * 10, hipMemcpyHostToDevice);

  // calc output gradients

  int osize = 0;
  int wsize = 0;

  int osize2 = 0;
  int wsize2 = 0;

  for (int i = 0; i < levels - 1; i++) {
    osize += m_levels[i].size();
  }

  if (levels > 2) {
    for (int i = 0; i < levels - 2; i++) {
      wsize += m_levels[i].size() * (m_levels[i + 1].size() - 1);
      osize2 += m_levels[i].size();
    }
  }

  wsize2 = wsize;

  dim3 dim_block(16, 1, 1);
  dim3 dim_grid(1, 1, 1);

  hipLaunchKernelGGL(calcOutGradientskernel, dim3(dim_grid), dim3(dim_block), 0,
                     0, goalVals_d, outputval_d, gradients_d, osize);
  hipDeviceSynchronize();

  // calc hidden gradients by going backwords through Network
  if (levels > 2) {

    for (int l = levels - 2; l > 0; --l) {

      dim3 dim_block(256, 1, 1);
      dim3 dim_grid((int)((m_levels[l].size() / 256) + 1), 1, 1);

      hipLaunchKernelGGL(calcHiddenGradientskernel, dim3(dim_grid),
                         dim3(dim_block), 0, 0, weights_d, gradients_d, osize2,
                         wsize2, topol_d, l, outputval_d);
      hipDeviceSynchronize();
      osize2 -= m_levels[l - 1].size();
      wsize2 -= m_levels[l - 1].size() * (m_levels[l].size() - 1);
    }
  }

  // update input weights
  for (int l = levels - 1; l > 0; --l) {

    dim3 dim_block(256, 1, 1);
    dim3 dim_grid((int)((m_levels[l].size() / 256) + 1), 1, 1);

    hipLaunchKernelGGL(updateInWeightskernel, dim3(dim_grid), dim3(dim_block),
                       0, 0, weights_d, gradients_d, outputval_d, wsize, osize,
                       derivWeights_d, topol_d, l);
    hipDeviceSynchronize();
    osize -= m_levels[l - 1].size();
    if (l - 2 >= 0)
      wsize -= m_levels[l - 2].size() * (m_levels[l - 1].size() - 1);
  }
}

void Network::backProp(const vector<double> &goalVals) {

  // calculate overall Network error (RMS of output Node errors)

  assert(goalVals.size() == m_levels.back().size() - 1);

  Level &outputLevel = m_levels.back();
  n_error = 0.0;

  for (unsigned n = 0; n < outputLevel.size() - 1; ++n) {
    double delta = goalVals[n] - outputLevel[n].getOutVal();
    n_error += delta * delta;
  }
  n_error /= outputLevel.size() - 1;
  n_error = sqrt(n_error);

  // Implement a recent average measurement

  m_AverageError = (m_AverageError * m_AverageSmoothingFactor + n_error) /
                   (m_AverageSmoothingFactor + 1.0);

  // Calculate output level gradients

  for (unsigned n = 0; n < outputLevel.size() - 1; ++n) {
    outputLevel[n].calcOutGradients(goalVals[n]);
  }

  // calculate gradients on all hidden levels

  for (unsigned levelNum = m_levels.size() - 2; levelNum > 0; --levelNum) {
    Level &hiddenLevel = m_levels[levelNum];
    Level &nextLevel = m_levels[levelNum + 1];

    for (unsigned n = 0; n < hiddenLevel.size(); ++n) {
      hiddenLevel[n].calcHiddenGradients(nextLevel);
    }
  }

  // From all levels from outputs to first hidden level, update Link weights

  for (unsigned levelNum = m_levels.size() - 1; levelNum > 0; --levelNum) {
    Level &level = m_levels[levelNum];
    Level &prevLevel = m_levels[levelNum - 1];

    for (unsigned n = 0; n < level.size() - 1; ++n) {
      level[n].updateInWeights(prevLevel);
    }
  }
}
