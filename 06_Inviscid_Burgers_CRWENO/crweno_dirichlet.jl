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
  x = Array{Float64}(undef, nx + 1)
  un = similar(x)  # numerical solsution at every time step
  ut = similar(x)  # temporary array during RK3 integration
  r = Array{Float64}(undef, nx)

  uL = Array{Float64}(undef, nx)
  uR = Array{Float64}(undef, nx + 1)

  a, b, c, d, e = (similar(x) for _ ∈ 1:5)

  k = 1  # record index
  freq = nt ÷ ns

  for i ∈ 1:nx + 1
    x[i] = Δx * (i - 1)
    un[i] = sin(2π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  # dirichlet boundary condition
  u[begin, k], u[end, k] = 0., 0.
  un[begin], un[end] = 0., 0.

  # dirichlet boundary condition for temporary array
  ut[begin], ut[end] = 0., 0.

  for j ∈ 2:nt + 1
    rhs(nx, Δx, un, r, uL, uR, a, b, c, d, e)

    @unroll ut[2:nx] = un[2:nx] + Δt * r[2:nx]

    rhs(nx, Δx, ut, r, uL, uR, a, b, c, d, e)

    @unroll ut[2:nx] = .75un[2:nx] + .25ut[2:nx] + .25Δt * r[2:nx]

    rhs(nx, Δx, ut, r, uL, uR, a, b, c, d, e)

    @unroll un[2:nx] = (1 / 3) * un[2:nx] + (2 / 3) * ut[2:nx] + (2 / 3) * Δt * r[2:nx]

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
rhs(nx, Δx, u, r, uL, uR, a, b, c, d, e) = begin
  crwenoL(nx, u, uL, a, b, c, d, e)
  crwenoR(nx, u, uR, a, b, c, d, e)

  @unroll r[2:nx] = -u[2:nx] * (
    u[2:nx] >= 0. ? uL[2:nx] - uL[1:nx-1] : uR[3:nx+1] - uR[2:nx]
  ) / Δx
  return
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for upwind direction (positive and left to right)
# u(i): solution values at finite difference grid nodes i = 0,1,...,N
# f(j): reconstructed values at nodes j = i+1/2; j = 0,1,...,N-1
# -----------------------------------------------------------------------------#
crwenoL(n, u, f, a, b, c, d, e, ε=1e-6) = begin
  i = 1
  b[i] = 2 / 3
  c[i] = 1 / 3
  d[i] = (u[i] + 5u[i+1]) / 6

  i = 2; a[i], b[i], c[i], b1, b2, b3 = crwcL(
    2u[i-1] - u[i],
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

  i = n
  a[i] = 1 / 3
  b[i] = 2 / 3
  d[i] = (5u[i] + u[i+1]) / 6

  tdms(a, b, c, d, f, e, 1, n)
  return
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative and right to left)
# u(i): solution values at finite difference grid nodes i = 0,1,...,N
# f(j): reconstructed values at nodes j = i-1/2; j = 1,2,...,N
# -----------------------------------------------------------------------------#
crwenoR(n, u, f, a, b, c, d, e, ε=1e-6) = begin
  i = 2
  b[i] = 2 / 3
  c[i] = 1 / 3
  d[i] = (u[i-1] + 5u[i]) / 6

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
    2u[i+1] - u[i], ε
  )
  d[i] = b1 * u[i-1] + b2 * u[i] + b3 * u[i+1]

  i = n + 1
  a[i] = 1 / 3
  b[i] = 2 / 3
  d[i] = (5u[i-1] + u[i]) / 6

  tdms(a, b, c, d, f, e, 2, n + 1)
  return
end

main() = begin
  for nx ∈ (100, 200, 400, 800, 1600)
    ns, Δt, tm = 10, .0001, .25

    Δx = 1. / nx
    nt = Int(tm / Δt)

    u = Array{Float64}(undef, nx + 1, ns)
    if boolenv("BENCH")
      @btime numerical($nx, $ns, $nt, $Δx, $Δt, $u)
    else
      @time numerical(nx, ns, nt, Δx, Δt, u)
    end
    x = Array(0:Δx:1.)

    open("solution_d_$nx.txt", "w") do io
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

# p1 = plot(x,u,lw = 1,
#           xlabel="\$X\$", ylabel = "\$U\$",
#           xlims=(minimum(x),maximum(x)),
#           grid=(:none), legend=:none)
#
# plot(p1, size = (1000, 600))
# savefig("crweno.pdf")
