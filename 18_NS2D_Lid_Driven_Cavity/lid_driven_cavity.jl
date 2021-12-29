include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils
using FFTW

# -----------------------------------------------------------------------------#
# Fast poisson solver for homozeneous Dirichlet domain
# -----------------------------------------------------------------------------#
fps_sine(nx, ny, Δx, Δy, f, u, e, data, data1, iden, sn) = begin
  @unroll data[1:nx-1, 1:ny-1] = f[2:nx, 2:ny]

  # y_k = 2 ∑[j=0->n-1] Xⱼ⋅sin(π(j+1)(k+1)/(n+1))
  e = FFTW.r2r(data, FFTW.RODFT00)

  @unroll data1[1:nx-1, 1:ny-1] = e[1:nx-1, 1:ny-1] * iden[1:nx-1, 1:ny-1]

  sn[2:nx, 2:ny] = FFTW.r2r(data1, FFTW.RODFT00) / ((2nx) * (2ny))
  return
end

"first order approximation"
bc(nx, ny, Δx, Δy, w, s) = begin
  # boundary condition for vorticity (Hoffmann) left and right
  @fastmath @simd for j ∈ 1:ny + 1
    w[1, j] = -2s[2, j] / Δx^2
    w[nx+1, j] = -2s[nx, j] / Δx^2
  end

  # boundary condition for vorticity (Hoffmann) bottom and top
  @fastmath @simd for i ∈ 1:nx + 1
    w[i, 1] = -2s[i, 2] / Δy^2
    w[i, ny+1] = -2s[i, ny] / Δy^2 - 2. / Δy
  end
end

"second order approximation"
bc2(nx, ny, Δx, Δy, w, s) = begin
  # boundary condition for vorticity (Jensen) left and right
  @fastmath @simd for j ∈ 1:ny + 1
    w[1, j] = (-4s[2, j] + .5s[3, j]) / Δx^2
    w[nx+1, j] = (-4s[nx, j] + .5s[nx-1, j]) / Δx^2
  end

  # boundary condition for vorticity (Jensen) bottom and top
  @fastmath @simd for i ∈ 1:nx + 1
    w[i, 1] = (-4s[i, 2] + .5s[i, 3]) / Δy^2
    w[i, ny+1] = (-4s[i, ny] + .5s[i, ny-1]) / Δy^2 - 3. / Δy
  end
end

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 2nd-order finite difference discretization
# -----------------------------------------------------------------------------#
numerical(nx, ny, nt, Δx, Δy, Δt, re, wn, sn, rms) = begin
  wt = Array{Float64}(undef, nx + 1, ny + 1)  # temporary array during RK3 integration
  r = similar(wt)  # right hand side
  sp = similar(wt)  # old streamfunction
  iden = similar(r)

  @fastmath @simd for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
    iden[i, j] = 1. / (
      (2 / Δx^2) * (cos(π * i / nx) - 1.) +
      (2 / Δy^2) * (cos(π * j / ny) - 1.)
    )
  end end

  u = Array{Complex{Float64}}(undef, nx - 1, ny - 1)
  e, data, data1 = (similar(u) for _ ∈ 1:3)

  for k ∈ 1:nt
    @unroll sp[1:nx+1, 1:ny+1] = sn[1:nx+1, 1:ny+1]  # previous timestep solution

    # Compute right-hand-side from vorticity
    rhs(nx, ny, Δx, Δy, re, wn, sn, r)

    @unroll wt[2:nx, 2:ny] = wn[2:nx, 2:ny] + Δt * r[2:nx, 2:ny]
    bc2(nx, ny, Δx, Δy, wt, sn)

    # compute streamfunction from vorticity
    fps_sine(nx, ny, Δx, Δy, -wt, u, e, data, data1, iden, sn)

    # Compute right-hand-side from vorticity
    rhs(nx, ny, Δx, Δy, re, wt, sn, r)

    @unroll wt[2:nx, 2:ny] = (
      .75wn[2:nx, 2:ny] +
      .25wt[2:nx, 2:ny] +
      .25Δt * r[2:nx, 2:ny]
    )
    bc2(nx, ny, Δx, Δy, wt, sn)

    # compute streamfunction from vorticity
    fps_sine(nx, ny, Δx, Δy, -wt, u, e, data, data1, iden, sn)

    # Compute right-hand-side from vorticity
    rhs(nx, ny, Δx, Δy, re, wt, sn, r)

    @unroll wn[2:nx, 2:ny] = (
      (1 / 3) * wn[2:nx, 2:ny] +
      (2 / 3) * wt[2:nx, 2:ny] +
      (2 / 3) * Δt * r[2:nx, 2:ny]
    )
    bc2(nx, ny, Δx, Δy, wn, sn)

    # compute streamfunction from vorticity
    fps_sine(nx, ny, Δx, Δy, -wn, u, e, data, data1, iden, sn)

    acc = 0.  # check for convergence (steaΔy state solution)
    @unroll acc += (sn[1:nx+1, 1:ny+1] - sp[1:nx+1, 1:ny+1])^2
    rms[k] = √(acc / ((nx + 1) * (ny + 1)))

    if mod(k, 100) == 0 println(k, " ", rms[k]) end
  end
end

# -----------------------------------------------------------------------------#
# Calculate right hand term of the inviscid Burgers equation
# r = -J(w,ψ) + ν ∇^2(w)
# -----------------------------------------------------------------------------#
rhs(nx, ny, Δx, Δy, re, w, s, r) = begin
  # Arakawa numerical scheme for Jacobian
  aa = 1 / (re * Δx^2)
  bb = 1 / (re * Δy^2)
  gg = 1 / (4Δx * Δy)
  hh = 1 / 3

  @fastmath @simd for j ∈ 2:ny for i ∈ 2:nx
    j1 = gg * (
      (w[i+1, j] - w[i-1, j]) * (s[i, j+1] - s[i, j-1]) -
      (w[i, j+1] - w[i, j-1]) * (s[i+1, j] - s[i-1, j])
    )

    j2 = gg * (
      w[i+1, j] * (s[i+1, j+1] - s[i+1, j-1]) -
      w[i-1, j] * (s[i-1, j+1] - s[i-1, j-1]) -
      w[i, j+1] * (s[i+1, j+1] - s[i-1, j+1]) +
      w[i, j-1] * (s[i+1, j-1] - s[i-1, j-1])
    )

    j3 = gg * (
      w[i+1, j+1] * (s[i, j+1] - s[i+1, j]) -
      w[i-1, j-1] * (s[i-1, j] - s[i, j-1]) -
      w[i-1, j+1] * (s[i, j+1] - s[i-1, j]) +
      w[i+1, j-1] * (s[i+1, j] - s[i, j-1])
    )

    jac = (j1 + j2 + j3) * hh

    # Central difference for Laplacian
    r[i, j] = -jac + (
      aa * (w[i+1, j] - 2w[i, j] + w[i-1, j]) +
      bb * (w[i, j+1] - 2w[i, j] + w[i, j-1])
    )
  end end
end

main() = begin
  nx, ny = 64, 64

  x_l, x_r = 0., 1.
  y_b, y_t = 0., 1.

  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  Δt, tf = .001, 10.
  nt = Int(tf / Δt)
  re = 100.

  x = Array{Float64}(undef, nx + 1)
  y = Array{Float64}(undef, ny + 1)
  rms = Array{Float64}(undef, nt)

  for i ∈ 1:nx + 1
    x[i] = Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = Δy * (i - 1)
  end

  wn = Array{Float64}(undef, nx + 1, ny + 1)
  sn = similar(wn)

  @unroll begin
    wn[1:nx+1, 1:ny+1] = 0.  # initial condition
    sn[1:nx+1, 1:ny+1] = 0.  # initial streamfunction
  end

  if boolenv("BENCH")
    @btime numerical($nx, $ny, $nt, $Δx, $Δy, $Δt, $re, $wn, $sn, $rms)

  else
    @time numerical(nx, ny, nt, Δx, Δy, Δt, re, wn, sn, rms)
  end

  t = Array(Δt:Δt:tf)

  open("res_plot.txt", "w") do io
    # write(io, "n res \n")
    for n ∈ 1:nt
      write(io, "$n $(rms[n])\n")
    end
  end

  open("field_final.txt", "w") do io
    # write(io, "x y wn sn \n")
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(wn[i, j]) $(sn[i, j])\n")
    end end
  end
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end