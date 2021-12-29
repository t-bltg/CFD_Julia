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
  un = similar(x)  # numerical solution at every time step
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

  # dirichlet boundary condition
  u[begin, k], u[end, k] = 0., 0.
  un[begin], un[end] = 0., 0.

  # dirichlet boundary condition for temporary array
  ut[begin], ut[end] = 0., 0.

  for j ∈ 1:nt
    rhs(nx, Δx, un, uL, uR, r)

    @unroll ut[2:nx] = un[2:nx] + Δt * r[2:nx]

    rhs(nx, Δx, ut, uL, uR, r)

    @unroll ut[2:nx] = .75un[2:nx] + .25ut[2:nx] + .25Δt * r[2:nx]

    rhs(nx, Δx, ut, uL, uR, r)

    @unroll un[2:nx] = (1 / 3) * un[2:nx] + (2 / 3) * ut[2:nx] + (2 / 3) * Δt * r[2:nx]

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
  wenoL_dirichlet(nx, u, uL)
  wenoR_dirichlet(nx, u, uR)

  @unroll r[2:nx] = -u[2:nx] * (
    u[2:nx] >= 0. ? uL[2:nx] - uL[1:nx-1] : uR[3:nx+1] - uR[2:nx]
  ) / Δx
  return
end

# -----------------------------------------------------------------------------#
# WENO reconstruction for upwind direction (positive; left to right)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# f(j): reconstructed values at nodes j = i+1/2; j = 1,...,N
# -----------------------------------------------------------------------------#
wenoL_dirichlet(n, u, f, ε=1e-6) = begin
  i = 1; f[i] = wcL(
    3u[i] - 2u[i+1],
    2u[i] - u[i+1],
    u[i],
    u[i+1],
    u[i+2], ε
  )

  i = 2; f[i] = wcL(
    2u[i-1] - u[i],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2], ε
  )

  for i ∈ 3:n - 1
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
    2u[i+1] - u[i], ε
  )
  return
end

# -----------------------------------------------------------------------------#
# CRWENO reconstruction for downwind direction (negative; right to left)
# u(i): solution values at finite difference grid nodes i = 1,...,N+1
# f(j): reconstructed values at nodes j = i-1/2; j = 2,...,N+1
# -----------------------------------------------------------------------------#
wenoR_dirichlet(n, u, f, ε=1e-6) = begin
  i = 2; f[i] = wcR(
    2u[i-1] - u[i],
    u[i-1],
    u[i],
    u[i+1],
    u[i+2],
    ε
  )

  for i ∈ 3:n - 1
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
    2u[i+1] - u[i], ε
  )

  i = n + 1; f[i] = wcR(
    u[i-2],
    u[i-1],
    u[i],
    2u[i] - u[i-1],
    3u[i] - 2u[i-1], ε
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

if (@__FILE__) ∈ ("string", abspath(PROGRAM_FILE))
  main()
end
