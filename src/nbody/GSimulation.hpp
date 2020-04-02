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
#define ALIGNMENT 64
#ifndef _GSIMULATION_HPP
#define _GSIMULATION_HPP

#include <iomanip>
#include <iostream>
#include <random>

#include "Particle.hpp"
#include "cpu_time.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

class GSimulation {
public:
  GSimulation();
  ~GSimulation();

  void init();
  void set_number_of_particles(int N);
  void set_number_of_steps(int N);
  void start();

  inline void set_cpu_ratio(const float &cpu_ratio) { _cpu_ratio = cpu_ratio; }
  inline void set_thread_dim0(const int &thread_dim0) {
    _thread_dim0 = thread_dim0;
  }
  inline void set_thread_dim1(const int &thread_dim1) {
    _thread_dim1 = thread_dim1;
  }
  inline int get_thread_dim0() { return _thread_dim0; }
  inline int get_thread_dim1() { return _thread_dim1; }
  inline int get_cpu_ratio() const { return _cpu_ratio; }
  inline void set_devices(int N) { _devices = N; };
  inline int get_devices() { return _devices; };

  int world_rank;
  int world_size;
  // int n; // number total particles
  int npp; // number perticles per process
  int *npp_global;
  void init_mpi();

private:
  ParticleSoA *particles;

  int _npart;       // number of particles
  int _nsteps;      // number of integration steps
  real_type _tstep; // time step of the simulation

  int _sfreq; // sample frequency

  real_type _kenergy; // kinetic energy

  double _totTime;  // total time of the simulation
  double _totFlops; // total number of flops

  float _cpu_ratio = -1.0f;
  int _thread_dim0 = 0;
  int _thread_dim1 = 0;
  int _devices = 0;

  void init_pos();
  void init_vel();
  void init_acc();
  void init_mass();

  CPUTime time;
  int nf;
  double ts0;
  double ts1;
  double nd;
  double gflops;
  double av = 0.0, dev = 0.0;
  int s;
  int nthreads = 1;

  inline void set_npart(const int &N) { _npart = N; }
  inline int get_npart() const { return _npart; }

  inline void set_tstep(const real_type &dt) { _tstep = dt; }
  inline real_type get_tstep() const { return _tstep; }

  inline void set_nsteps(const int &n) { _nsteps = n; }
  inline int get_nsteps() const { return _nsteps; }

  inline void set_sfreq(const int &sf) { _sfreq = sf; }
  inline int get_sfreq() const { return _sfreq; }

  void print_header();
  void print_stats();
  void print_flops();

  void mpi_bcast_all();
  void mpi_gather_acc(int start);
};

#endif
