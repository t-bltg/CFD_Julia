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
  un = similar(x)  # numerical solution at every time step
  ut = similar(x)  # temporary array during RK3 integration

  # flux computed at nodal points and positive and negative splitting
  r, f, fP, fN = (similar(x) for _ ∈ 1:4)

  # wave speed at nodal points
  ps = similar(x)

  # left and right side fluxes at the interface
  fL = Array{Float64}(undef, nx + 1)
  fR = similar(fL)

  k = 1  # record index
  freq = nt ÷ ns

  for i ∈ 1:nx
    x[i] = -.5Δx + Δx * i
    un[i] = sin(2π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  # TVD RK3 for time integration
  for n ∈ 1:nt  # time step
    rhs(nx, Δx, un, r, f, fP, fN, fL, fR, ps)

    @unroll ut[1:nx] = un[1:nx] + Δt * r[1:nx]

    rhs(nx, Δx, ut, r, f, fP, fN, fL, fR, ps)

    @unroll ut[1:nx] = .75un[1:nx] + .25ut[1:nx] + .25Δt * r[1:nx]

    rhs(nx, Δx, ut, r, f, fP, fN, fL, fR, ps)

    @unroll un[1:nx] = un[1:nx] / 3 + (2 / 3) * ut[1:nx] + (2 / 3) * Δt * r[1:nx]

    if mod(n, freq) == 0
      @show n
      k += 1
      @unroll u[1:nx, k] = un[1:nx]
    end
  end
  return
end

# -----------------------------------------------------------------------------#
# Calculate right hand side terms of the Euler equations
# -----------------------------------------------------------------------------#
rhs(nx, Δx, u, r, f, fP, fN, fL, fR, ps) = begin
  @unroll f[1:nx] = .5u[1:nx]^2

  wavespeed(nx, u, ps)

  @unroll begin
    fP[1:nx] = .5(f[1:nx] + ps[1:nx] * u[1:nx])
    fN[1:nx] = .5(f[1:nx] - ps[1:nx] * u[1:nx])
  end

  # WENO Reconstruction
  # compute upwind reconstruction for positive flux (left to right)
  wenoL(nx, fP, fL)
  # compute downwind reconstruction for negative flux (right to left)
  wenoR(nx, fN, fR)

  # compute RHS using flux splitting
  @unroll r[1:nx] = -(fL[2:nx+1] - fL[1:nx]) / Δx - (fR[2:nx+1] - fR[1:nx]) / Δx
  return
end

# -----------------------------------------------------------------------------#
# Compute wave speed (Jacobian = df/du)
# -----------------------------------------------------------------------------#
wavespeed(n, u, ps) = begin
  @unroll ps[3:n-2] = max(
    abs(u[1:n-4]), abs(u[2:n-3]),
    abs(u[3:n-2]),
    abs(u[4:n-1]), abs(u[5:n])
  )
  # periodicity
  @fastmath for i ∈ (1, 2, n - 1, n)
    ps[i] = max(
      abs(u[circ(i-2, n)]),
      abs(u[circ(i-1, n)]),
      abs(u[i]),
      abs(u[circ(i+1, n)]),
      abs(u[circ(i+2, n)])
    )
  end
end

main() = begin
  nx, ns = 150, 10
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