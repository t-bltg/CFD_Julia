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
numerical(nx, ns, nt, Δx, Δt, q) = begin
  x = Array{Float64}(undef, nx)
  r = Array{Float64}(undef, nx, 3)
  qn = similar(r)  # numerical solsution at every time step
  qt = similar(r)  # temporary array during RK3 integration

  ri = 1  # record index
  freq = nt ÷ ns

  γ = 1.4  # specific gas ratio

  # Sod's Riemann problem
  ρL, uL, pL = 1., 0., 1.  # Left side
  ρR, uR, pR = .125, 0., .1  # Right side

  # nodal storage location (grid)
  for i ∈ 1:nx
    x[i] = -.5Δx + i * Δx
  end

  xc = .5  # seperator location
  for i ∈ 1:nx
    ρ, u, p = x[i] > xc ? (ρR, uR, pR) : (ρL, uL, pL)
    e = p / (ρ * (γ - 1.)) + .5u^2

    # conservative variables
    qn[i, 1] = ρ
    qn[i, 2] = ρ * u
    qn[i, 3] = ρ * e
  end

  for m ∈ 1:3
    @unroll q[1:nx, m, ri] = qn[1:nx, m]  # store solution at t=0
  end

  f = Array{Float64}(undef, nx + 1, 3)
  qL, qR, fL, fR = (similar(f) for _ ∈ 1:4)

  # TVD RK3 for time integration
  for n ∈ 1:nt  # time step
    rhs(nx, Δx, γ, qn, r, qL, qR, fL, fR, f)

    for m ∈ 1:3
      @unroll qt[1:nx, m] = qn[1:nx, m] + Δt * r[1:nx, m]
    end

    rhs(nx, Δx, γ, qt, r, qL, qR, fL, fR, f)

    for m ∈ 1:3
      @unroll qt[1:nx, m] = .75qn[1:nx, m] + .25qt[1:nx, m] + .25Δt * r[1:nx, m]
    end

    rhs(nx, Δx, γ, qt, r, qL, qR, fL, fR, f)

    for m ∈ 1:3
      @unroll qn[1:nx, m] = (1 / 3) * qn[1:nx, m] + (2 / 3) * qt[1:nx, m] + (2 / 3) * Δt * r[1:nx, m]
    end

    if mod(n, freq) == 0
      @show n
      ri += 1
      for m ∈ 1:3
        @unroll q[1:nx, m, ri] = qn[1:nx, m]
      end
    end
  end
  return
end

# -----------------------------------------------------------------------------#
# Calculate right hand side terms of the Euler equations
# -----------------------------------------------------------------------------#
rhs(nx, Δx, γ, q, r, qL, qR, fL, fR, f) = begin
  # WENO Reconstruction
  wenoL_roe(nx, q, qL)
  wenoR_roe(nx, q, qR)

  # Computing fluxes
  fluxes_roe(nx, γ, qL, fL)
  fluxes_roe(nx, γ, qR, fR)

  # compute Riemann solver using HLLC scheme
  hllc(nx, γ, qL, qR, f, fL, fR)

  # RHS
  for m ∈ 1:3
    @unroll r[1:nx, m] = -(f[2:nx+1, m] - f[1:nx, m]) / Δx
  end
end

# -----------------------------------------------------------------------------#
# Riemann solver: HLLC
# -----------------------------------------------------------------------------#
hllc(nx, γ, uL, uR, f, fL, fR) = begin
  gm = γ - 1.
  Ds = Float64[0., 1., 0.]

  @fastmath @simd for i ∈ 1:nx + 1
    # left state
    rhLL = uL[i, 1]
    uuLL = uL[i, 2] / rhLL
    eeLL = uL[i, 3] / rhLL
    ppLL = gm * (eeLL * rhLL - .5rhLL * uuLL^2)
    aaLL = √(abs(γ * ppLL / rhLL))

    # right state
    rhRR = uR[i, 1]
    uuRR = uR[i, 2] / rhRR
    eeRR = uR[i, 3] / rhRR
    ppRR = gm * (eeRR * rhRR - .5rhRR * uuRR^2)
    aaRR = √(abs(γ * ppRR / rhRR))

    # compute SL and Sr
    SL = min(uuLL, uuRR) - max(aaLL, aaRR)
    SR = max(uuLL, uuRR) + max(aaLL, aaRR)

    # compute compound speed
    Ds[3] = SP = (
      (ppRR - ppLL + rhLL * uuLL * (SL - uuLL) - rhRR * uuRR * (SR - uuRR)) /
      (rhLL * (SL - uuLL) - rhRR * (SR - uuRR))  # never get zero
    )

    # compute compound pressure
    PLR = (
      .5(ppLL + ppRR + rhLL * (SL - uuLL) * (SP - uuLL) + rhRR * (SR - uuRR) * (SP - uuRR))
    )

    for m ∈ 1:3
      f[i, m] = (
        SL >= 0. ? fL[i, m] : (
          SR <= 0. ? fR[i, m] : (
            (SP >= 0.) & (SL <= 0.) ? (SP * (SL * uL[i, m] - fL[i, m]) + SL * PLR * Ds[m]) / (SL - SP) : (
              (SP <= 0.) & (SR >= 0.) ? (SP * (SR * uR[i, m] - fR[i, m]) + SR * PLR * Ds[m]) / (SR - SP) :
              f[i, m]
            )
          )
        )
      )
    end
  end
end

main() = begin
  nx, ns = 8192, 20
  Δt, tm = .00005, .2

  Δx = 1. / nx
  nt = Int(tm / Δt)

  q = zeros(Float64, nx, 3, ns + 1)
  if boolenv("BENCH")
    @btime numerical($nx, $ns, $nt, $Δx, $Δt, $q)
  else
    @time numerical(nx, ns, nt, Δx, Δt, q)
  end

  x = Array(.5Δx:Δx:1. - .5Δx)

  open("solution_dF.txt", "w") do iod
    open("solution_vF.txt", "w") do iov
      open("solution_eF.txt", "w") do ioe
        for i ∈ 1:nx
          write(iod, "$(x[i]) ")
          write(iov, "$(x[i]) ")
          write(ioe, "$(x[i]) ")
          for n ∈ 1:ns + 1
            write(iod, "$(q[i, 1, n]) ")
            write(iov, "$(q[i, 2, n]) ")
            write(ioe, "$(q[i, 3, n]) ")
          end
          write(iod, "\n")
          write(iov, "\n")
          write(ioe, "\n")
        end
      end
    end
  end
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

#####################
# OBSOLETE/COMMENTS #
#####################

#=
# original
if (SL >= 0.)
  for m = 1:3
    f[i, m] = fL[i, m]
  end
elseif (SR <= 0.)
  for m = 1:3
    f[i, m] = fR[i, m]
  end
elseif ((SP >= 0.) & (SL <= 0.))
  for m = 1:3
    f[i, m] = (SP * (SL * uL[i, m] - fL[i, m]) + SL * PLR * Ds[m]) / (SL - SP)
  end
elseif ((SP <= 0.) & (SR >= 0.))
  for m = 1:3
    f[i, m] = (SP * (SR * uR[i, m] - fR[i, m]) + SR * PLR * Ds[m]) / (SR - SP)
  end
end
=#

#=
# very slow
@views begin
  if (SL >= 0.)
    f[i, :] = fL[i, :]
  elseif (SR <= 0.)
    f[i, :] = fR[i, :]
  elseif ((SP >= 0.) & (SL <= 0.))
    f[i, :] = (SP * (SL * uL[i, :] - fL[i, :]) + SL * PLR * Ds) / (SL - SP)
  elseif ((SP <= 0.) & (SR >= 0.))
    f[i, :] = (SP * (SR * uR[i, :] - fR[i, :]) + SR * PLR * Ds) / (SR - SP)
  end
end
=#