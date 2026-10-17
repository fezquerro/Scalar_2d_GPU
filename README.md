# Monte Carlo Massive Scalar field GPU Simulator (2+1)

**A high-performance lattice simulation code for GPU-accelerated Monte Carlo simulation of a massive scalar field in 2+1 dimensions using CUDA.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)

## Overview

This project implements a Monte Carlo simulation for computing the energy of a massive scalar field in 2+1 dimensions with different boundary conditions and compare with the analytical results. The code uses **CUDA** for NVIDIA GPUs and is parallelized at the maximum extend allowed by the action of a massive scalar field. Five different boundary conditions are implemented for the second spatial dimension. 

The analytical results that this code compare its results to are presented in the PhD thesis, **[Properties of the quantum vacuum in non-abelian gauge theories](https://arxiv.org/abs/2605.18220)**, in Chapter 3.

## Features

- **Efficient updates**: checkerboard Metropolis, fully implemented in CUDA kernels
- **GPU Acceleration**: High-performance CUDA implementation
- **Flexible lattice sizes**: Support for different $N_t \times N_x \times N_y$ lattices and masses $m$
- **Boundary conditions**: Periodic, Anti-periodic, Dirichlet, Neumann and Zaremba boundary conditions
- **Observables**:
  - Energy (first component of the energy-momentum tensor)
- **Analytical results**: The code compares the compatibility of the simulation results with the analytical formulas
- **Thermalization and autocorrelation analysis**
- **Configurable random number generator** (with reproducible seeds, CUDA-compatible): MRG32k3a (one stored state per site) or Philox4_32_10 (counter based, no stored state, faster), selected with the `RNG` flag in `kernel.cu`
