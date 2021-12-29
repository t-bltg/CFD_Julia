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
  qn = similar(r)  # numerical solution at every time step
  qt = similar(r)  # temporary array during RK3 integration

  f = Array{Float64}(undef, nx + 1, 3)
  qL, qR, fL, fR, ps = (similar(f) for _ ∈ 1:5)

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
  xc = .5  # separator location

  for i ∈ 1:nx
    ρ, u, p = x[i] > xc ? (ρR, uR, pR) : (ρL, uL, pL)
    e = p / (ρ * (γ - 1)) + .5u^2

    # conservative variables
    # qn[i, :] = ρ, ρ * u, ρ * e  # Not a valid syntax
    qn[i, 1] = ρ
    qn[i, 2] = ρ * u
    qn[i, 3] = ρ * e
  end

  for m ∈ 1:3
    @unroll q[1:nx, m, ri] = qn[1:nx, m]  # store solution at t=0
  end

  # TVD RK3 for time integration
  for n ∈ 1:nt  # time step
    rhs(nx, Δx, γ, qn, r, f, qL, qR, fL, fR, ps)

    for m ∈ 1:3
      @unroll qt[1:nx, m] = qn[1:nx, m] + Δt * r[1:nx, m]
    end

    rhs(nx, Δx, γ, qt, r, f, qL, qR, fL, fR, ps)

    for m ∈ 1:3
      @unroll qt[1:nx, m] = .75qn[1:nx, m] + .25qt[1:nx, m] + .25Δt * r[1:nx, m]
    end

    rhs(nx, Δx, γ, qt, r, f, qL, qR, fL, fR, ps)

    for m ∈ 1:3
      @unroll qn[1:nx, m] = (
        (1. / 3.) * qn[1:nx, m] + (2. / 3.) * qt[1:nx, m] + (2. / 3.) * Δt * r[1:nx, m]
      )
    end

    if mod(n, freq) == 0
      @show n
      ri += 1
      for m ∈ 1:3
        @unroll q[1:nx, m, ri] = qn[1:nx, m]
      end
    end
  end
end

# -----------------------------------------------------------------------------#
# Calculate right hand side terms of the Euler equations
# -----------------------------------------------------------------------------#
rhs(nx, Δx, γ, q, r, f, qL, qR, fL, fR, ps) = begin
  # WENO Reconstruction
  wenoL_roe(nx, q, qL)
  wenoR_roe(nx, q, qR)

  # Computing fluxes
  fluxes_roe(nx, γ, qL, fL)
  fluxes_roe(nx, γ, qR, fR)

  # compute Riemann solver using Gudanov scheme
  gudanov(nx, γ, qL, qR, f, fL, fR, ps)

  # RHS
  for m ∈ 1:3
    @unroll r[1:nx, m] = -(f[2:nx+1, m] - f[1:nx, m]) / Δx
  end
end

# -----------------------------------------------------------------------------#
# Riemann solver: Rusanov
# -----------------------------------------------------------------------------#
gudanov(nx, γ, qL, qR, f, fL, fR, ps) = begin
  # wavespeed2(nx, γ, q, ps)
  wavespeed(nx, γ, qL, qR, ps)
  # Interface fluxes (Rusanov)
  for m ∈ 1:3
    @unroll f[1:nx+1, m] = (
      .5(fR[1:nx+1, m] + fL[1:nx+1, m]) -
      .5ps[1:nx+1] * (qR[1:nx+1, m] - qL[1:nx+1, m])
    )
  end
end

# -----------------------------------------------------------------------------#
# compute wave speed
# -----------------------------------------------------------------------------#
wavespeed2(nx, γ, q, ps) = begin
  rad = Array{Float64}(undef, nx)
  # spectral radius of Jacobian
  @fastmath @simd for i ∈ 1:nx
    a = √(γ * ((γ - 1.) * (q[i, 3] - .5q[i, 2] * q[i, 2] / q[i, 1])) / q[i, 1])
    rad[i] = max(
      abs(q[i, 2] / q[i, 1]),
      abs(q[i, 2] / q[i, 1] + a),
      abs(q[i, 2] / q[i, 1] - a)
    )
  end

  # propagation speed
  @unroll ps[2:nx] = max(rad[1:nx-1], rad[2:nx])
  ps[1] = ps[2]
  ps[nx+1] = ps[nx]
  return
end

wavespeed(nx, γ, uL, uR, ps) = begin
  rad = Array{Float64}(undef, nx)
  # spectral radius of Jacobian
  gm = γ - 1.
  @fastmath @simd for i ∈ 1:nx + 1
    # Left and right states:
    rhLL = uL[i, 1]
    uuLL = uL[i, 2] / rhLL
    eeLL = uL[i, 3] / rhLL
    ppLL = gm * (eeLL * rhLL - .5rhLL * uuLL^2)
    hhLL = eeLL + ppLL / rhLL

    rhRR = uR[i, 1]
    uuRR = uR[i, 2] / rhRR
    eeRR = uR[i, 3] / rhRR
    ppRR = gm * (eeRR * rhRR - .5rhRR * uuRR^2)
    hhRR = eeRR + ppRR / rhRR

    α = 1. / (√(abs(rhLL)) + √(abs(rhRR)))

    uu = (√(abs(rhLL)) * uuLL + √(abs(rhRR)) * uuRR) * α
    hh = (√(abs(rhLL)) * hhLL + √(abs(rhRR)) * hhRR) * α
    aa = √(abs(gm * (hh - .5uu^2)))

    # ps[i] = aa + abs(uu)
    ps[i] = abs(aa + uu)
  end
end

main() = begin
  nx, ns = 8192, 20
  Δt, tm = .00005, .2

  Δx = 1. / nx
  nt = Int(tm / Δt)

  # q = Array{Float64}(zero, nx,3,ns+1)
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
