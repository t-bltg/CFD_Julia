include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Printf
using Utils

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 5th-order Compact WENO scheme for spatial terms
# -----------------------------------------------------------------------------#
numerical(nx, ns, nt, Δx, Δt, u) = begin
  x = Array{Float64}(undef, nx)
  un = similar(x)  # numerical solsution at every time step
  ut = similar(x)  # temporary array during RK3 integration
  r = similar(x)

  # flux used to compute right hand side
  f = Array{Float64}(undef, nx + 1)
  uL, uR, fL, fR, ps = (similar(f) for _ ∈ 1:5)

  k = 1  # record index
  freq = nt ÷ ns

  for i ∈ 1:nx
    x[i] = -.5Δx + Δx * (i)
    un[i] = sin(2π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  # TVD RK3 for time integration
  for n ∈ 1:nt  # time step
    rhs(nx, Δx, un, r, uL, uR, fL, fR, f, ps)

    @unroll ut[1:nx] = un[1:nx] + Δt * r[1:nx]

    rhs(nx, Δx, ut, r, uL, uR, fL, fR, f, ps)

    @unroll ut[1:nx] = .75un[1:nx] + .25ut[1:nx] + .25Δt * r[1:nx]

    rhs(nx, Δx, ut, r, uL, uR, fL, fR, f, ps)

    @unroll un[1:nx] = (1 / 3) * un[1:nx] + (2 / 3) * ut[1:nx] + (2 / 3) * Δt * r[1:nx]

    if mod(n, freq) == 0
      @show n * Δt
      k += 1
      @unroll u[1:nx, k] = un[1:nx]
    end
  end
  return
end

# -----------------------------------------------------------------------------#
# Calculate fluxes
# -----------------------------------------------------------------------------#
fluxes(nx, u, f) = begin
  @unroll f[1:nx+1] = .5u[1:nx+1]^2
  return
end

# -----------------------------------------------------------------------------#
# Calculate right hand side terms of the Euler equations
# -----------------------------------------------------------------------------#
rhs(nx, Δx, u, r, uL, uR, fL, fR, f, ps) = begin
  # WENO Reconstruction
  wenoL(nx, u, uL)
  wenoR(nx, u, uR)

  # Computing fluxes
  fluxes(nx, uL, fL)
  fluxes(nx, uR, fR)

  # compute Riemann solver (flux at interface)
  rusanov(nx, u, uL, uR, f, fL, fR, ps)

  # RHS
  @unroll r[1:nx] = -(f[2:nx+1] - f[1:nx]) / Δx
  return
end

# -----------------------------------------------------------------------------#
# Riemann solver: Rusanov
# -----------------------------------------------------------------------------#
rusanov(nx, u, uL, uR, f, fL, fR, ps) = begin
  @unroll ps[2:nx] = max(abs(u[2:nx]), abs(u[1:nx-1]))
  ps[1] = max(abs(u[1]), abs(u[nx]))
  ps[nx+1] = max(abs(u[1]), abs(u[nx]))

  # Interface fluxes (Rusanov)
  @unroll f[1:nx+1] = (
    .5(fR[1:nx+1] + fL[1:nx+1]) -
    .5ps[1:nx+1] * (uR[1:nx+1] - uL[1:nx+1])
  )
  return
end


main() = begin
  nx, ns = 200, 10
  Δt, tm = .0001, .25

  Δx = 1. / nx
  nt = Int(tm / Δt)

  u = Array{Float64}(undef, nx, ns + 1)

  if boolenv("BENCH")
    @btime numerical($nx, $ns, $nt, $Δx, $Δt, $u)
  else
    @time numerical(nx, ns, nt, Δx, Δt, u)
  end

  x = Array(.5Δx:Δx:1 - .5Δx)

  open("solution_p.txt", "w") do io
    for i ∈ 1:nx
      write(io, "$(x[i]) ")
      for j ∈ 1:ns
        write(io, "$(u[i, j]) ")
      end
      write(io, "\n")
    end
  end
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end