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

#include "GSimulation.hpp"
#include "cpu_time.hpp"

GSimulation ::GSimulation() {
  set_npart(2000);
  set_nsteps(500);
  set_tstep(0.1);
  set_sfreq(50);
  init_mpi();
  if (world_rank == 0) {
    std::cout << "===============================" << std::endl;
    std::cout << " Initialize Gravity Simulation" << std::endl;
  }
}

void GSimulation ::set_number_of_particles(int N) { set_npart(N); }

void GSimulation ::set_number_of_steps(int N) { set_nsteps(N); }

void GSimulation ::init_pos() {
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
  }
}

void GSimulation ::init_vel() {
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation ::init_acc() {
  for (int i = 0; i < get_npart(); ++i) {
    particles->acc_x[i] = 0.f;
    particles->acc_y[i] = 0.f;
    particles->acc_z[i] = 0.f;
  }
}

void GSimulation ::init_mass() {
  real_type n = static_cast<real_type>(get_npart());
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles->mass[i] = n * unif_d(gen);
  }
}

void GSimulation ::init_mpi() {
#ifdef USE_MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int n = get_npart();
  npp_global = (int *)malloc(world_size * sizeof(int));
  if (world_rank == 0) {
    npp = n / world_size + n % world_size;
    npp_global[0] = npp;
    for (int i = 1; i < world_size; i++)
      npp_global[i] = n / world_size;
  } else {
    npp = n / world_size;
  }
  MPI_Bcast(npp_global, world_size, MPI_INT, 0, MPI_COMM_WORLD);
  // std::cout << "Rank: " << world_rank << " Share: " << npp << std::endl;
#endif
}

void GSimulation ::print_header() {
  if (world_rank == 0) {
    std::cout << " nPart = " << get_npart() << "; "
              << "nSteps = " << get_nsteps() << "; "
              << "dt = " << get_tstep() << std::endl;

    std::cout << "------------------------------------------------"
              << std::endl;
    std::cout << " " << std::left << std::setw(8) << "s" << std::left
              << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
              << std::left << std::setw(12) << "time (s)" << std::left
              << std::setw(12) << "GFlops" << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;
  }
}

void GSimulation ::print_stats() {
  if (world_rank == 0) {
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
  }
}

void GSimulation ::print_flops() {
  if (world_rank == 0) {
    std::cout << std::endl;
    std::cout << "# Number Threads     : " << nthreads << std::endl;
    std::cout << "# Total Time (s)     : " << _totTime << std::endl;
    std::cout << "# Average Perfomance : " << av << " +- " << dev << std::endl;
    std::cout << "===============================" << std::endl;
  }
}

void GSimulation ::mpi_bcast_all() {
#ifdef USE_MPI
  int n = get_npart();
  // update all ranks with latest data from master
  MPI_Bcast(particles->vel_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->vel_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->vel_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast(particles->pos_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->pos_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->pos_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast(particles->acc_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->acc_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(particles->acc_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void GSimulation ::mpi_gather_acc(int start) {
#ifdef USE_MPI
  int n = get_npart();
  float accx[n];
  float accy[n];
  float accz[n];
  int disp[world_size];
  disp[0] = 0;
  for (int i = 1; i < world_size; i++)
    disp[i] = disp[i - 1] + npp_global[i - 1];

  MPI_Gatherv(particles->acc_x + disp[world_rank], npp_global[world_rank],
              MPI_FLOAT, accx, npp_global, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(particles->acc_y + disp[world_rank], npp_global[world_rank],
              MPI_FLOAT, accy, npp_global, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(particles->acc_z + disp[world_rank], npp_global[world_rank],
              MPI_FLOAT, accz, npp_global, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int ii = 0; ii < n; ii++) {
    particles->acc_x[ii] = accx[ii];
    particles->acc_y[ii] = accy[ii];
    particles->acc_z[ii] = accz[ii];
  }
#endif
}

GSimulation ::~GSimulation() {
  free(particles->pos_x);
  free(particles->pos_y);
  free(particles->pos_z);
  free(particles->vel_x);
  free(particles->vel_y);
  free(particles->vel_z);
  free(particles->acc_x);
  free(particles->acc_y);
  free(particles->acc_z);
  free(particles->mass);
  free(particles);

#ifdef USE_MPI
  MPI_Finalize();
#endif
}
