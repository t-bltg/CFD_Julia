#!/usr/bin/env bash
{
  . activate jl || die
  set -e -x
  add_pid() {
    pids+=($1)
    if true; then wait $1; fi
  }
  fd=/dev/stdout
  # fd=/dev/null

  alias julia='jl-lts-deps'

  (cd 01_Heat_Equation_FTCS; julia ftcs.jl  1>$fd) & add_pid $!
  (cd 02_Heat_Equation_RK3; julia rk3.jl    1>$fd) & add_pid $!
  (cd 03_Heat_Equation_CN; julia cn.jl      1>$fd) & add_pid $!
  (cd 04_Heat_Equation_ICP; julia icp.jl    1>$fd) & add_pid $!
  (
    cd 05_Inviscid_Burgers_WENO
    julia weno_trial.jl     1>$fd
    julia weno_dirichlet.jl 1>$fd
    julia weno_periodic.jl  1>$fd
  ) & add_pid $!
  (
    cd 06_Inviscid_Burgers_CRWENO
    julia crweno_periodic.jl  1>$fd
    julia crweno_dirichlet.jl 1>$fd
  ) & add_pid $!
  (cd 07_Inviscid_Burgers_Flux_Splitting; julia burgers_flux_splitting.jl 1>$fd) & add_pid $!
  (cd 08_Inviscid_Burgers_Rieman; julia burgers_riemann.jl 1>$fd) & add_pid $!
  (cd 09_Euler_1D_Roe; julia euler_roe.jl   1>$fd) & add_pid $!
  (cd 10_Euler_1D_HLLC; julia euler_hllc.jl 1>$fd) & add_pid $!
  (cd 11_Euler_1D_Rusanov; julia euler_rusanov.jl 1>$fd) & add_pid $!
  (cd 12_Poisson_Solver_FFT; julia fft_p.jl 1>$fd) & add_pid $!
  (cd 13_Poisson_Solver_FFT_Spectral; julia fft_s.jl 1>$fd) & add_pid $!
  (cd 14_Poisson_Solver_FST; julia fft_d.jl 1>$fd) & add_pid $!
  (cd 15_Poisson_Solver_Gauss_Seidel; julia gauss_seidel.jl 1>$fd) & add_pid $!  # very long
  (cd 16_Poisson_Solver_Conjugate_Gradient; julia conjugate_gradient.jl 1>$fd) & add_pid $!
  (
    cd 17_Poisson_Solver_Multigrid
    julia mg_N.jl 1>$fd
    julia mg.jl   1>$fd
  ) & add_pid $!
  (cd 18_NS2D_Lid_Driven_Cavity; julia lid_driven_cavity.jl 1>$fd) & add_pid $!
  (
    cd 19_NS2D_Vortex_Merger
    julia tgv.jl  1>$fd
    julia vm.jl   1>$fd
  ) & add_pid $!
  (cd 20_NS2D_Hybrid_Solver; julia hybrid.jl 1>$fd) & add_pid $!
  (cd 21_NS2D_PseudoSpectral_32_Rule; julia pseudospectral_32_rule.jl 1>$fd) & add_pid $!
  (cd 22_NS2D_PseudoSpectral_23_Rule; julia pseudospectral_23_rule.jl 1>$fd) & add_pid $!

  eval trap \"kill -TERM ${pids[@]}\" INT
  wait ${pids[@]}
  exit
}