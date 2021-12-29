include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils
using Plots

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 2nd-order finite difference discretization
# -----------------------------------------------------------------------------#
numerical(nx, ny, nt, Δx, Δy, Δt, re, wn) = begin
  wt = Array{Float64}(undef, nx + 2, ny + 2)  # temporary array during RK3 integration
  r, s = (similar(wt) for _ ∈ 1:2)

  f = Array{Float64}(undef, nx, ny)

  u = Array{Complex{Float64}}(undef, nx, ny)
  e, data, data1 = (similar(u) for _ ∈ 1:3)

  for k ∈ 1:nt
    if mod(k, 100) == 0 println(k) end
    # Compute right-hand-side from vorticity
    vm_rhs(nx, ny, Δx, Δy, re, wn, u, e, data, data1, r, s, f)

    @unroll wt[2:nx+1, 2:ny+1] = wn[2:nx+1, 2:ny+1] + Δt * r[2:nx+1, 2:ny+1]

    @views begin
      # periodic BC
      wt[nx+2, :] = wt[2, :]
      wt[:, ny+2] = wt[:, 2]

      # ghost points
      wt[1, :] = wt[nx+1, :]
      wt[:, 1] = wt[:, ny+1]
    end

    # Compute right-hand-side from vorticity
    vm_rhs(nx, ny, Δx, Δy, re, wt, u, e, data, data1, r, s, f)

    @unroll wt[2:nx+1, 2:ny+1] = (
      .75wn[2:nx+1, 2:ny+1] +
      .25wt[2:nx+1, 2:ny+1] +
      .25Δt * r[2:nx+1, 2:ny+1]
    )

    @views begin
      # periodic BC
      wt[nx+2, :] = wt[2, :]
      wt[:, ny+2] = wt[:, 2]

      # ghost points
      wt[1, :] = wt[nx+1, :]
      wt[:, 1] = wt[:, ny+1]
    end

    # Compute right-hand-side from vorticity
    vm_rhs(nx, ny, Δx, Δy, re, wt, u, e, data, data1, r, s, f)

    @unroll wn[2:nx+1, 2:ny+1] = (
      wn[2:nx+1, 2:ny+1] / 3. +
      (2 / 3) * wt[2:nx+1, 2:ny+1] +
      (2 / 3) * Δt * r[2:nx+1, 2:ny+1]
    )

    @views begin
      # periodic BC
      wn[nx+2, :] = wn[2, :]
      wn[:, ny+2] = wn[:, 2]

      # ghost points
      wn[1, :] = wn[nx+1, :]
      wn[:, 1] = wn[:, ny+1]
    end
  end

  return wn[2:nx+2, 2:ny+2]
end

# compute exact solution for TGV problem
exact_tgv(nx, ny, x, y, time, re) = begin
  ue = Array{Float64}(undef, nx + 1, ny + 1)

  nq = 4
  @fastmath @simd for i ∈ 1:nx + 1 for j ∈ 1:ny + 1
    ue[i, j] = 2nq * cos(nq * x[i]) * cos(nq * y[j]) * exp(-2nq^2 * time / re)
  end end
  return ue
end

main() = begin
  nx, ny = 64, 64

  x_l, x_r = 0., 2π
  y_b, y_t = 0., 2π

  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  Δt, tf = .01, 1.
  nt = Int(tf / Δt)
  re = 10.

  x = Array{Float64}(undef, nx + 1)
  y = Array{Float64}(undef, ny + 1)

  for i ∈ 1:nx + 1
    x[i] = Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = Δy * (i - 1)
  end

  wn = Array{Float64}(undef, nx + 2, ny + 2)

  wn[2:nx+2, 2:ny+2] = exact_tgv(nx, ny, x, y, 0., re)

  @views begin
    # ghost points
    wn[1, :] = wn[nx+1, :]
    wn[:, 1] = wn[:, ny+1]
  end

  val, t, bytes, gctime, memallocs = @timed begin
    un = numerical(nx, ny, nt, Δx, Δy, Δt, re, wn)
  end

  println("CPU Time=$t")

  ue = exact_tgv(nx, ny, x, y, tf, re)

  uerror = un - ue
  rms_error = compute_l2norm_bnds(nx, ny, uerror)
  max_error = maximum(abs.(uerror))

  println("Error details:")
  println("L-2 Norm=$rms_error")
  println("Maximum Norm=$max_error")

  p1 = contour(x, y, transpose(ue), fill=true, xlabel="\$X\$", ylabel="\$Y\$", title="Exact")
  p2 = contour(x, y, transpose(un), fill=true, xlabel="\$X\$", ylabel="\$Y\$", title="Numerical")
  p3 = plot(p1, p2, size=(1300, 600))
  savefig(p3, "tgv.pdf")
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end