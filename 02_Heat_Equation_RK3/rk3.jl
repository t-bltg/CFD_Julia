include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Printf
using Plots
using Utils

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 5th-order Compact WENO scheme for spatial terms
# -----------------------------------------------------------------------------#
numerical(nx, nt, Δx, Δt, x, u, α) = begin
  un = Array{Float64}(undef, nx + 1)  # numerical solsution at every time step
  ut = Array{Float64}(undef, nx + 1)  # temporary array during RK3 integration
  r = Array{Float64}(undef, nx)

  k = 1  # record index

  for i ∈ 1:nx + 1
    un[i] = -sin(π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  # dirichlet boundary condition
  un[begin], un[end] = 0., 0.

  # dirichlet boundary condition for temporary array
  ut[begin], ut[end] = 0., 0.

  for j ∈ 2:nt + 1
    rhs(nx, Δx, un, r, α)

    @unroll ut[2:nx] = un[2:nx] + Δt * r[2:nx]

    rhs(nx, Δx, ut, r, α)

    @unroll ut[2:nx] = .75un[2:nx] + .25ut[2:nx] + .25Δt * r[2:nx]

    rhs(nx, Δx, ut, r, α)

    @unroll un[2:nx] = un[2:nx] / 3. + (2 / 3) * ut[2:nx] + (2 / 3) * Δt * r[2:nx]

    k += 1
    u[:, k] = un
  end
  return
end

# -----------------------------------------------------------------------------#
# Calculate right hand term of the inviscid Burgers equation
# r = -u∂u/∂x
# -----------------------------------------------------------------------------#
rhs(nx, Δx, u, r, α) = begin
  @unroll r[2:nx] = α * (u[3:nx+1] - 2u[2:nx] + u[1:nx-1]) / Δx^2
  return
end

main() = begin
  x_l, x_r = -1., 1.
  Δx = .025
  nx = Int((x_r - x_l) / Δx)

  Δt = .0025
  t = 1.
  nt = Int(t / Δt)

  α = 1. / π^2

  u = Array{Float64}(undef, nx + 1, nt + 1)
  x = Array{Float64}(undef, nx + 1)
  u_e = similar(x)

  for i ∈ 1:nx + 1
    x[i] = x_l + Δx * (i - 1)  # location of each grid point
    u_e[i] = -exp(-t) * sin(π * x[i])  # initial condition @ t=0
  end

  if boolenv("BENCH")
    @btime numerical($nx, $nt, $Δx, $Δt, $x, $u, $α)
  else
    @time numerical(nx, nt, Δx, Δt, x, u, α)
  end

  u_n = u[:, nt+1]
  u_error = u_n - u_e
  rms_error = compute_l2norm(nx, u_error)

  # create output file for L2-norm
  open("output.txt", "w") do f
    write(f, "Error details:\n")
    write(f, "L-2 Norm=$rms_error\n")
    write(f, "Maximum Norm=$(maximum(abs.(u_error)))\n")
  end

  # create text file for final field
  open("field_final.csv", "w") do f
    write(f, "x ue un uerror\n")
    for i ∈ 1:nx + 1
      write(f, "$(x[i]) $(u_e[i]) $(u_n[i]) $(u_error[i])\n")
    end
  end
  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end