#include "Network.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

vector<double> inVals;
vector<double> resultVals;
vector<double> goalVals;

Network myNetwork;

int reverseInt(int i) {
  unsigned char x1, x2, x3, x4;

  x1 = i & 255;
  x2 = (i >> 8) & 255;
  x3 = (i >> 16) & 255;
  x4 = (i >> 24) & 255;

  return ((int)x1 << 24) + ((int)x2 << 16) + ((int)x3 << 8) + x4;
}

double *tvals = new double[10];

void train_on_gpu(int amount) {

  ifstream file("train-images-idx3-ubyte");
  ifstream file2("train-labels-idx1-ubyte");

  int magic_n = 0;
  int n_of_images = 0;
  int n_rows = 0;
  int n_columns = 0;
  file.read((char *)&magic_n, sizeof(magic_n));
  magic_n = reverseInt(magic_n);
  file.read((char *)&n_of_images, sizeof(n_of_images));
  n_of_images = reverseInt(n_of_images);
  file.read((char *)&n_rows, sizeof(n_rows));
  n_rows = reverseInt(n_rows);
  file.read((char *)&n_columns, sizeof(n_columns));
  n_columns = reverseInt(n_columns);
  printf("Total number of images=%d ; size of each image: #rows=%d #cols=%d\n",
         n_of_images, n_rows, n_columns);

  int magic_n2 = 0;
  int n_of_images2 = 0;

  file2.read((char *)&magic_n2, sizeof(magic_n2));
  magic_n2 = reverseInt(magic_n2);

  file2.read((char *)&n_of_images2, sizeof(n_of_images2));
  n_of_images2 = reverseInt(n_of_images2);
  printf("Total number of labels=%d\n", n_of_images2);

  double *invals = new double[784];

  for (int i = 0; i < amount; ++i) {
    if ((i % (amount / 10)) == 0)
      printf("progress: %.0f%%\n", i * 100.0f / amount);
    for (int r = 0; r < 28; ++r) {
      for (int c = 0; c < 28; ++c) {
        unsigned char temp = 0;

        file.read((char *)&temp, sizeof(temp));

        double in = ((double)(int)temp) / 255.0;

        in *= 2.0;
        in -= 1.0;

        invals[(28 * r) + c] = in;
      }
    }

    myNetwork.FdFwdParallel(invals);

    unsigned char label = 0;
    file2.read((char *)&label, sizeof(label));

    for (int x = 0; x < 10; x++) {

      tvals[x] = -1.0;
    }

    tvals[(int)label] = 1.0;

    myNetwork.backPropParallel(tvals);
  }
}

int test_on_gpu() {

  ifstream train("t10k-images-idx3-ubyte");
  ifstream trainlabel("t10k-labels-idx1-ubyte");

  int magic_n = 0;
  int n_of_images = 0;
  int n_rows = 0;
  int n_columns = 0;
  int magic_n2 = 0;
  int n_of_images2 = 0;
  int error = 0;

  train.read((char *)&magic_n, sizeof(magic_n));
  magic_n = reverseInt(magic_n);
  train.read((char *)&n_of_images, sizeof(n_of_images));
  n_of_images = reverseInt(n_of_images);
  train.read((char *)&n_rows, sizeof(n_rows));
  n_rows = reverseInt(n_rows);
  train.read((char *)&n_columns, sizeof(n_columns));
  n_columns = reverseInt(n_columns);

  trainlabel.read((char *)&magic_n2, sizeof(magic_n2));
  magic_n2 = reverseInt(magic_n2);

  trainlabel.read((char *)&n_of_images2, sizeof(n_of_images2));
  n_of_images2 = reverseInt(n_of_images2);

  double *invals = new double[784];

  for (int i = 0; i < 10000; ++i) {
    for (int r = 0; r < 28; ++r) {
      for (int c = 0; c < 28; ++c) {
        unsigned char temp = 0;

        train.read((char *)&temp, sizeof(temp));

        double in = ((double)(int)temp) / 255.0;

        in *= 2.0;
        in -= 1.0;

        invals[(28 * r) + c] = in;
      }
    }

    myNetwork.FdFwdParallel(invals);
    myNetwork.getResultsFromGPU();

    unsigned char label = 0;
    trainlabel.read((char *)&label, sizeof(label));

    // max result
    double maxr = myNetwork.results_h[0];
    int maxindex = 0;
    for (int x = 1; x < 10; x++) {

      if (myNetwork.results_h[x] > maxr) {
        maxr = myNetwork.results_h[x];
        maxindex = x;
      }
    }

    if (((int)label) != maxindex) {
      error++;
    }
  }
  return error;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  vector<unsigned> topol;

  // network topology
  topol.push_back(784);
  topol.push_back(100);
  topol.push_back(10);

  myNetwork.init(topol);

  // train and test using 10,000 images using a GPU
  myNetwork.allocmemGPU();
  train_on_gpu(10000);
  double error = ((double)test_on_gpu()) / 10000.0 * 100;
  cout << "Error rate: " << error << "%" << endl;
  myNetwork.copyGpuToCpu();
  // myNetwork.outputToFile("Networks/784-100-10");

  myNetwork.deallocmemGPU();
  cout << "DONE" << endl;
  return 0;
}
