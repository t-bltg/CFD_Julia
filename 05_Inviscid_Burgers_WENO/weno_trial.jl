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
      # println(j * Δt)
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
  # wenoL(nx, u, uL)  # not used here (simple FTCS instead, for comparing with WENO)
  # wenoR(nx, u, uR)
  @unroll r[2:nx] = -u[2:nx] * (u[3:nx+1] - u[1:nx-1]) / (2. * Δx)
  return
end

main() = begin
  for nx ∈ (100, 200, 400)
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

    open("solution_$nx.txt", "w") do io
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

#####################
# OBSOLETE/COMMENTS #
#####################

# if ()
#   # r[i] = -(u[i+1]^2 - u[i-1]^2)/(2.*Δx)
# else
#   r[i] = -u[i] * () / (2. * Δx)
#   # r[i] = -(u[i+1]^2 - u[i-1]^2)/(2.*Δx)
# end
