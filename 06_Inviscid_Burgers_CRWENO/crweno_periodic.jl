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
numerical(nx, ns, nt, dx, dt, u) = begin
  x = Array{Float64}(undef, nx + 1)
  un = similar(x)  # numerical solution at every time step
  ut = similar(x)  # temporary array during RK3 integration
  r = Array{Float64}(undef, nx)

  uL = Array{Float64}(undef, nx)
  uR = Array{Float64}(undef, nx + 1)
  a, b, c, d, e, f, g, h = (similar(x) for _ ∈ 1:8)

  k = 1  # record index
  freq = nt ÷ ns

  for i ∈ 1:nx + 1
    x[i] = dx * (i - 1)
    un[i] = sin(2π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  for j ∈ 2:nt + 1
    rhs(nx, dx, un, r, uL, uR, a, b, c, d, e, f, g, h)

    @unroll ut[1:nx] = un[1:nx] + dt * r[1:nx]
    ut[end] = ut[begin]  # periodic

    rhs(nx, dx, ut, r, uL, uR, a, b, c, d, e, f, g, h)

    @unroll ut[1:nx] = .75un[1:nx] + .25ut[1:nx] + .25dt * r[1:nx]
    ut[end] = ut[begin]  # periodic

    rhs(nx, dx, ut, r, uL, uR, a, b, c, d, e, f, g, h)

    @unroll un[1:nx] = (1 / 3) * un[1:nx] + (2 / 3) * ut[1:nx] + (2 / 3) * dt * r[1:nx]
    un[end] = un[begin]  # periodic

    if mod(j, freq) == 0
      u[:, k] = un
      k += 1
    end
  end
end

# -----------------------------------------------------------------------------#
# Calculate right hand term of the inviscid Burgers equation
# r = -u⋅∂u/∂x
# -----------------------------------------------------------------------------#
rhs(nx, dx, u, r, uL, uR, a, b, c, d, e, f, g, h) = begin
  crwenoL(nx, u, uL, a, b, c, d, e, f, g, h)
  crwenoR(nx, u, uR, a, b, c, d, e, f, g, h)

  @unroll r[2:nx] = -u[2:nx] * (
    u[2:nx] >= 0. ? uL[2:nx] - uL[1:nx-1] : uR[3:nx+1] - uR[2:nx]
  ) / dx
  for i ∈ (1,)  # periodic
    r[i] = -u[i] * (u[i] >= 0. ? uL[i] - uL[nx] : uR[i+1] - uR[nx+1]) / dx
  end
end

# -----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
# -----------------------------------------------------------------------------#
ctdms(a, b, c, α, β, r, x, k, u, w, l, s, e) = begin
  γ = -b[s]
  k[s] -= γ
  k[e] -= α * β / γ

  @unroll k[s+1:e-1] = b[s+1:e-1]

  tdms(a, k, c, r, x, l, s, e)

  u[s] = γ
  u[e] = α
  @unroll u[s+1:e-1] = 0.

  tdms(a, k, c, u, w, l, s, e)

  fact = (x[s] + β * x[e] / γ) / (1. + w[s] + β * w[e] / γ)

  @unroll x[s:e] -= fact * w[s:e]
  return
end


# -----------------------------------------------------------------------------#
# CRWENO reconstruction for upwind direction (positive; left to right)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# m(j): reconstructed values at nodes j <== i+1/2; only use j = 1,2,...,N
# -----------------------------------------------------------------------------#
crwenoL(n, u, m, a, b, c, d, e, f, g, h, ε=1e-6) = begin
  i = 1; a[i], b[i], c[i], b1, b2, b3 = crwcL(
    u[n-1],
    u[n],
    u[i],
    u[i+1],
    u[i+2], ε
  )
  d[i] = b1 * u[n] + b2 * u[i] + b3 * u[i+1]

  i = 2; a[i], b[i], c[i], b1, b2, b3 = crwcL(
    u[n],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]

  @fastmath @simd for i ∈ 3:n - 1
    a[i], b[i], c[i], b1, b2, b3 = crwcL(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )
    d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]
  end

  i = n; a[i], b[i], c[i], b1, b2, b3 = crwcL(
    u[i-2],
    u[i-1],
    u[i],
    u[i+1],
    u[2], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]

  ss, ee = 1, n
  ctdms(a, b, c, c[ee], a[ss], d, m, e, f, g, h, ss, ee)
  return
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative;right to left)
# u(i): solution values at finite difference grid nodes i =1,...,N+1
# m(j): reconstructed values at nodes j <== i-1/2; only use j = 2,...,N+1
# -----------------------------------------------------------------------------#
crwenoR(n, u, m, a, b, c, d, e, f, g, h, ε=1e-6) = begin
  i = 2; a[i], b[i], c[i], b1, b2, b3 = crwcR(
    u[n],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]

  @fastmath @simd for i ∈ 3:n - 1
    a[i], b[i], c[i], b1, b2, b3 = crwcR(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )
    d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]
  end

  i = n; a[i], b[i], c[i], b1, b2, b3 = crwcR(
    u[i-2],
    u[i-1],
    u[i],
    u[i+1],
    u[2], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]

  i = n + 1; a[i], b[i], c[i], b1, b2, b3 = crwcR(
    u[i-2],
    u[i-1],
    u[i],
    u[2],
    u[3], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[2]

  ss, ee = 2, n + 1
  ctdms(a, b, c, c[ee], a[ss], d, m, e, f, g, h, ss, ee)
  return
end

main() = begin
  for nx ∈ (100, 200, 400, 800, 1600)
    ns, dt, tm = 10, .0001, .25

    dx = 1. / nx
    nt = Int(tm / dt)

    u = Array{Float64}(undef, nx + 1, ns + 1)
    if boolenv("BENCH")
      @btime numerical($nx, $ns, $nt, $dx, $dt, $u)
    else
      @time numerical(nx, ns, nt, dx, dt, u)
    end
    x = Array(0:dx:1.)

    open("solution_p_$nx.txt", "w") do io
      for i ∈ 1:nx + 1
        write(io, "$(x[i]) ")
        for j ∈ 1:ns
          write(io, "$(u[i, j]) ")
        end
        write(io, "\n")
      end
    end
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end