include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils

mg(Δx, Δy, nx, ny, r, f, u_n, v1, v2, v3, max_iter, tol, out, freq=100) = begin
  # create text file for writing residual history
  res = open("residual.txt", "w")
  # write(res_plot, "k"," ","rms"," ","rms/rms0"," \n")

  # compute initial residual
  compute_residual(nx, ny, Δx, Δy, f, u_n, r)
  # compute initial L-2 norm
  init_rms = rms = compute_l2norm(nx, ny, r)

  println("0 $rms $(rms / init_rms)")

  # allocate memory for grid size at different levels
  lnx = zeros(Int, 2)
  lny = zero(lnx)
  lΔx = zeros(Float64, 2)
  lΔy = zero(lΔx)
  lnx[1], lny[1] = nx, ny
  lnx[2] = lnx[1] ÷ 2
  lny[2] = lny[1] ÷ 2
  lΔx[1], lΔy[1] = Δx, Δy
  lΔx[2] = lΔx[1] * 2
  lΔy[2] = lΔy[1] * 2

  # allocate matrix for storage at fine level
  # residual at fine level is alreaΔy defined at global level
  prol_fine = zeros(Float64, lnx[1] + 1, lny[1] + 1)

  # allocate matrix for storage at coarse levels
  fc = zeros(Float64, lnx[2] + 1, lny[2] + 1)
  unc = zero(fc)

  # start main iteration loop
  it = 0
  while it < max_iter
    it += 1
    # call relaxation on fine grid and compute the numerical solution for fixed number of iterations
    gauss_seidel_mg(lnx[1], lny[1], Δx, Δy, f, u_n, v1)

    # check for convergence only for finest grid, compute the residual and L2 norm
    compute_residual(nx, ny, Δx, Δy, f, u_n, r)

    # compute the l2norm of residual
    rms = compute_l2norm(nx, ny, r)

    if mod(it, freq) == 0
      # write results only for finest residual
      write(res, "$it $rms $(rms / init_rms)\n")
      println("$it $rms $(rms / init_rms)")
    end

    if rms / init_rms <= tol break end

    # restrict the residual from fine level to coarse level
    restriction(lnx[1], lny[1], lnx[2], lny[2], r, fc)

    fill!(unc, 0.)  # set solution zero on coarse grid

    # solve on the coarsest level and relax V3 times
    gauss_seidel_mg(lnx[2], lny[2], lΔx[2], lΔy[2], fc, unc, v3)

    # V-cycle, denscend the solution
    # prolongate solution from coarse level to fine level
    prolongation(lnx[2], lny[2], lnx[1], lny[1], unc, prol_fine)

    # correct the solution on fine level
    @simd for j ∈ 2:lny[1] for i ∈ 2:lnx[1]
      u_n[i, j] += prol_fine[i, j]
    end end

    gauss_seidel_mg(lnx[1], lny[1], Δx, Δy, f, u_n, v2)  # relax v2 times
  end

  write(out, "L-2 Norm=$rms\n")
  write(out, "Maximum Norm=$(maximum(abs.(r)))\n")
  write(out, "Iterations=$it\n")
  close(res)
  return
end

main() = begin
  nx, ny = 256, 256
  max_iter = 100_000
  tol = 1e-9

  # create output file for L2-norm
  out = open("output.txt", "w")
  write(out, "Residual details:\n")

  x_l, x_r = 0., 1.
  y_b, y_t = 0., 1.

  v1 = 2  # relaxation
  v2 = 2  # prolongation
  v3 = 2  # coarsest level
  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  # allocate array for x and y position of grids, exact solution and source term
  x = Array{Float64}(undef, nx + 1)
  y = Array{Float64}(undef, ny + 1)
  f = Array{Float64}(undef, nx + 1, ny + 1)
  u_e = similar(f)
  u_n = similar(f)

  for i ∈ 1:nx + 1
    x[i] = x_l + Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = y_b + Δy * (i - 1)
  end

  c1 = (1. / 16.)^2
  c2 = -2π^2

  for i ∈ 1:nx + 1 for j ∈ 1:ny + 1
    u_e[i, j] = sin(2π * x[i]) * sin(2π * y[j]) + c1 * sin(16π * x[i]) * sin(16π * y[j])

    f[i, j] = 4c2 * sin(2π * x[i]) * sin(2π * y[j]) + c2 * sin(16π * x[i]) * sin(16π * y[j])

    u_n[i, j] = 0.
  end end

  @views begin
    u_n[:, 1] = u_e[:, 1]
    u_n[:, ny+1] = u_e[:, ny+1]

    u_n[1, :] = u_e[1, :]
    u_n[nx+1, :] = u_e[nx+1, :]
  end

  r = zeros(Float64, nx + 1, ny + 1)

  open("field_initial.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(u_n[i, j]) $(u_e[i, j])\n")
    end end
  end
  val, t, bytes, gctime, memallocs = @timed begin
    mg(Δx, Δy, nx, ny, r, f, u_n, v1, v2, v3, max_iter, tol, out)
  end

  u_error = u_n - u_e
  rms_error = compute_l2norm(nx, ny, u_error)
  max_error = maximum(abs.(u_error))

  write(out, "Error details:\n")
  write(out, "L-2 Norm=$rms_error\n")
  write(out, "Maximum Norm=$max_error\n")
  write(out, "CPU Time=$t\n")

  open("field_final.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(u_n[i, j]) $(u_e[i, j])\n")
    end end
  end

  close(out)
  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
