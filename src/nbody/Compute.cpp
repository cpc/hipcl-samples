/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".

    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdlib.h>

#include "hip/hip_runtime.h"
#include "GSimulation.hpp"
#include "cpu_time.hpp"

__global__ void nbody(real_type *particles_pos_x, real_type *particles_pos_y,
                      real_type *particles_pos_z, real_type *particles_acc_x,
                      real_type *particles_acc_y, real_type *particles_acc_z,
                      real_type *particles_mass, const int n) {
  const float softeningSquared = 1.e-3f;
  const float G = 6.67259e-11f;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    real_type ax_i = 0;
    real_type ay_i = 0;
    real_type az_i = 0;

    for (int j = 0; j < n; j++) {
      real_type dx, dy, dz;
      real_type distanceSqr = 0.0f;
      real_type distanceInv = 0.0f;

      dx = particles_pos_x[j] - particles_pos_x[i]; // 1flop
      dy = particles_pos_y[j] - particles_pos_y[i]; // 1flop
      dz = particles_pos_z[j] - particles_pos_z[i]; // 1flop

      distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared; // 6flops
      // distanceInv = 1.0f / __fsqrt_rz(distanceSqr); //1div+1sqrt
      distanceInv = 1.0f / sqrtf(distanceSqr); // 1div+1sqrt

      ax_i += dx * G * particles_mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
      ay_i += dy * G * particles_mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
      az_i += dz * G * particles_mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
    }
    particles_acc_x[i] = ax_i;
    particles_acc_y[i] = ay_i;
    particles_acc_z[i] = az_i;
  }
}

void GSimulation ::start() {
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  const int alignment = 32;
  particles = (ParticleSoA *)aligned_alloc(alignment, sizeof(ParticleSoA));

  particles->pos_x =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->pos_y =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->pos_z =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->vel_x =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->vel_y =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->vel_z =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->acc_x =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->acc_y =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->acc_z =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));
  particles->mass =
      (real_type *)aligned_alloc(alignment, n * sizeof(real_type));

  real_type *particles_pos_x_d;
  real_type *particles_pos_y_d;
  real_type *particles_pos_z_d;

  real_type *particles_acc_x_d;
  real_type *particles_acc_y_d;
  real_type *particles_acc_z_d;

  real_type *particles_mass_d;

  hipMalloc((void **)&particles_pos_x_d, n * sizeof(real_type));
  hipMalloc((void **)&particles_pos_y_d, n * sizeof(real_type));
  hipMalloc((void **)&particles_pos_z_d, n * sizeof(real_type));

  hipMalloc((void **)&particles_acc_x_d, n * sizeof(real_type));
  hipMalloc((void **)&particles_acc_y_d, n * sizeof(real_type));
  hipMalloc((void **)&particles_acc_z_d, n * sizeof(real_type));

  hipMalloc((void **)&particles_mass_d, n * sizeof(real_type));

  init_pos();
  init_vel();
  init_acc();
  init_mass();

  hipMemcpy(particles_mass_d, particles->mass, n * sizeof(real_type),
            hipMemcpyHostToDevice);

  hipMemcpy(particles_pos_x_d, particles->pos_x, n * sizeof(real_type),
            hipMemcpyHostToDevice);
  hipMemcpy(particles_pos_y_d, particles->pos_y, n * sizeof(real_type),
            hipMemcpyHostToDevice);
  hipMemcpy(particles_pos_z_d, particles->pos_z, n * sizeof(real_type),
            hipMemcpyHostToDevice);

  hipMemcpy(particles_acc_x_d, particles->acc_x, n * sizeof(real_type),
            hipMemcpyHostToDevice);
  hipMemcpy(particles_acc_y_d, particles->acc_y, n * sizeof(real_type),
            hipMemcpyHostToDevice);
  hipMemcpy(particles_acc_z_d, particles->acc_z, n * sizeof(real_type),
            hipMemcpyHostToDevice);

  print_header();

  _totTime = 0.;

  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ((11. + 18.) * nd * nd + nd * 19.);
  double av = 0.0, dev = 0.0;
  int nf = 0;

  size_t block_size;
  if (get_thread_dim0() != 0) {
    block_size = get_thread_dim0();
  } else {
    block_size = 256;
  }
  const size_t grid_size = (n + block_size - 1) / block_size;
  std::cout << "using block_size = " << block_size << std::endl;

  const double t0 = time.start();
  for (int s = 1; s <= get_nsteps(); ++s) {

    ts0 += time.start();

    hipMemcpy(particles_pos_x_d, particles->pos_x, n * sizeof(real_type),
              hipMemcpyHostToDevice);
    hipMemcpy(particles_pos_y_d, particles->pos_y, n * sizeof(real_type),
              hipMemcpyHostToDevice);
    hipMemcpy(particles_pos_z_d, particles->pos_z, n * sizeof(real_type),
              hipMemcpyHostToDevice);

    hipLaunchKernelGGL(nbody, dim3(grid_size), dim3(block_size), 0, 0,
                       particles_pos_x_d, particles_pos_y_d, particles_pos_z_d,
                       particles_acc_x_d, particles_acc_y_d, particles_acc_z_d,
                       particles_mass_d, n);

    hipMemcpy(particles->acc_x, particles_acc_x_d, n * sizeof(real_type),
              hipMemcpyDeviceToHost);
    hipMemcpy(particles->acc_y, particles_acc_y_d, n * sizeof(real_type),
              hipMemcpyDeviceToHost);
    hipMemcpy(particles->acc_z, particles_acc_z_d, n * sizeof(real_type),
              hipMemcpyDeviceToHost);

    energy = 0;

    for (int i = 0; i < n; ++i) // update position
    {
      particles->vel_x[i] += particles->acc_x[i] * dt; // 2flops
      particles->vel_y[i] += particles->acc_y[i] * dt; // 2flops
      particles->vel_z[i] += particles->acc_z[i] * dt; // 2flops

      particles->pos_x[i] += particles->vel_x[i] * dt; // 2flops
      particles->pos_y[i] += particles->vel_y[i] * dt; // 2flops
      particles->pos_z[i] += particles->vel_z[i] * dt; // 2flops

      //     no need since OCL overwrites
      particles->acc_x[i] = 0.;
      particles->acc_y[i] = 0.;
      particles->acc_z[i] = 0.;

      energy += particles->mass[i] *
                (particles->vel_x[i] * particles->vel_x[i] +
                 particles->vel_y[i] * particles->vel_y[i] +
                 particles->vel_z[i] * particles->vel_z[i]); // 7flops
    }

    _kenergy = 0.5 * energy;

    ts1 += time.stop();
    if (!(s % get_sfreq())) {
      nf += 1;
      std::cout << " " << std::left << std::setw(8) << s << std::left
                << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(5) << std::setw(12)
                << _kenergy << std::left << std::setprecision(5)
                << std::setw(12) << (ts1 - ts0) << std::left
                << std::setprecision(5) << std::setw(12)
                << gflops * get_sfreq() / (ts1 - ts0) << std::endl;
      if (nf > 2) {
        av += gflops * get_sfreq() / (ts1 - ts0);
        dev += gflops * get_sfreq() * gflops * get_sfreq() /
               ((ts1 - ts0) * (ts1 - ts0));
      }

      ts0 = 0;
      ts1 = 0;
    }

  } // end of the time step loop

  const double t1 = time.stop();
  _totTime = (t1 - t0);
  _totFlops = gflops * get_nsteps();

  av /= (double)(nf - 2);
  dev = sqrt(dev / (double)(nf - 2) - av * av);

  int nthreads = 1;

  std::cout << std::endl;
  std::cout << "# Number Threads     : " << nthreads << std::endl;
  std::cout << "# Total Time (s)     : " << _totTime << std::endl;
  std::cout << "# Average Perfomance : " << av << " +- " << dev << std::endl;
  std::cout << "===============================" << std::endl;
}
