include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
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

  k = 1  # record index
  freq = nt ÷ ns

  for i ∈ 1:nx + 1
    x[i] = Δx * (i - 1)
    un[i] = sin(2π * x[i])
    u[i, k] = un[i]  # store solution at t=0
  end

  for j ∈ 2:nt + 1
    rhs(nx, Δx, un, uL, uR, r)

    @unroll ut[1:nx] = un[1:nx] + Δt * r[1:nx]
    ut[end] = ut[begin]  # periodic

    rhs(nx, Δx, ut, uL, uR, r)

    @unroll ut[1:nx] = .75un[1:nx] + .25ut[1:nx] + .25Δt * r[1:nx]
    ut[end] = ut[begin]  # periodic

    rhs(nx, Δx, ut, uL, uR, r)

    @unroll un[1:nx] = (1 / 3) * un[1:nx] + (2 / 3) * ut[1:nx] + (2 / 3) * Δt * r[1:nx]
    un[end] = un[begin]  # periodic

    if mod(j, freq) == 0
      u[:, k] = un
      k += 1
    end
  end
  return
end

# -----------------------------------------------------------------------------#
# Calculate right hand term of the inviscid Burgers equation
# r = -u⋅∂u/∂x
# -----------------------------------------------------------------------------#
rhs(nx, Δx, u, uL, uR, r) = begin
  crwenoL(nx, u, uL)
  crwenoR(nx, u, uR)

  @unroll r[2:nx] = -u[2:nx] * (
    u[2:nx] >= 0. ? uL[2:nx] - uL[1:nx-1] : uR[3:nx+1] - uR[2:nx]
  ) / Δx
  for i ∈ 1:1  # periodic
    r[i] = -u[i] * (u[i] >= 0. ? uL[i] - uL[nx] : uR[i+1] - uR[nx+1]) / Δx
  end
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for upwind direction (positive; left to right)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# f(j): reconstructed values at nodes j <== i+1/2; only use j = 1,2,...,N
# -----------------------------------------------------------------------------#
crwenoL(n, u, f, ε=1e-6) = begin
  i = 1; f[i] = wcL(
    u[n-1],
    u[n],
    u[i],
    u[i+1],
    u[i+2], ε
  )

  i = 2; f[i] = wcL(
    u[n],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2], ε
  )

  @simd for i ∈ 3:n - 1
    f[i] = wcL(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )
  end

  i = n; f[i] = wcL(
    u[i-2],
    u[i-1],
    u[i],
    u[i+1],
    u[2], ε
  )
  return
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative;right to left)
# u(i): solution values at finite difference grid nodes i =1,...,N+1
# f(j): reconstructed values at nodes j <== i-1/2; only use j = 2,...,N+1
# -----------------------------------------------------------------------------#
crwenoR(n, u, f, ε=1e-6) = begin
  i = 2; f[i] = wcR(
    u[n],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2], ε
  )

  @simd for i ∈ 3:n - 1
    f[i] = wcR(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )
  end

  i = n; f[i] = wcR(
    u[i-2],
    u[i-1],
    u[i],
    u[i+1],
    u[2], ε
  )

  i = n + 1; f[i] = wcR(
    u[i-2],
    u[i-1],
    u[i],
    u[2],
    u[3], ε
  )
  return
end

main() = begin
  for nx ∈ (100, 200, 400)
    ns, Δt, tm = 10, .0001, .25

    Δx = 1. / nx
    nt = Int(tm / Δt)

    u = Array{Float64}(undef, nx + 1, ns + 1)
    if boolenv("BENCH")
      @btime numerical($nx, $ns, $nt, $Δx, $Δt, $u)
    else
      @time numerical(nx, ns, nt, Δx, Δt, u)
    end

    x = Array(0:Δx:1.)

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