include("../Common.jl")
using .Common
using BenchmarkTools
using DelimitedFiles
using Unroll
using Printf

gauss_seidel(Δx, Δy, nx, ny, r, f, u_n, max_iter, tol, out, freq=10_000) = begin
  # create text file for writing residual history
  res = open("gs_residual.txt", "w")
  # write(res, "k"," ","rms"," ","rms/rms0"," \n")

  compute_residual(nx, ny, Δx, Δy, f, u_n, r)
  init_rms = rms = compute_l2norm(nx, ny, r)

  println("0 $rms $(rms / init_rms)")

  it = 0
  while it < 5max_iter
    it += 1
    #=
    @show @macroexpand @fastmath @simd for j = 2:ny for i = 2:nx
      r[i, j] = f[i, j] - (
        (u_n[i+1, j] - 2. * u_n[i, j] + u_n[i-1, j]) / Δx^2 +
        (u_n[i, j+1] - 2. * u_n[i, j] + u_n[i, j-1]) / Δy^2
      )
    end end
    exit(0)
    =#

    # compute solution at next time step ϕ^(k+1) = ϕ^k + ωr^(k+1)
    # residual = f + λ^2u - ∇^2u
    @fastmath @simd for j ∈ 2:ny for i ∈ 2:nx
      r[i, j] = f[i, j] - (
        (u_n[i+1, j] - 2u_n[i, j] + u_n[i-1, j]) / Δx^2 +
        (u_n[i, j+1] - 2u_n[i, j] + u_n[i, j-1]) / Δy^2
      )
    end end
    @unroll u_n[2:nx, 2:ny] += r[2:nx, 2:ny] / (-2 / Δx^2 - 2 / Δy^2)

    if mod(it, freq) == 0
      compute_residual(nx, ny, Δx, Δy, f, u_n, r)
      rms = compute_l2norm(nx, ny, r)
      write(res, "$it $rms $(rms / init_rms)\n")
      println("$it $rms $(rms / init_rms)")
      if (rms / init_rms) <= tol break end
    end
  end

  write(out, "L-2 Norm=$rms\n")
  write(out, "Maximum Norm=$(maximum(abs.(r)))\n")
  write(out, "Iterations=$it\n")
  close(res)
end

main() = begin
  nx, ny = 512, 512
  max_iter = 20 * 100_000
  tol = 1e-9

  # create output file for L2-norm
  out = open("output.txt", "w")
  write(out, "Residual details:\n")

  # create text file for initial and final field
  initial = open("field_initial.txt", "w")
  final = open("field_final.txt", "w")

  # write(field_initial, "x y f un ue \n")
  # write(field_final, "x y f un ue e \n")
  ipr = 1

  x_l, x_r = 0., 1.
  y_b, y_t = 0., 1.

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

  c1 = (1 / 16)^2
  c2 = -2π^2

  @simd for i ∈ 1:nx + 1 for j ∈ 1:ny + 1
    if ipr == 1
      u_e[i, j] = (x[i]^2 - 1.) * (y[j]^2 - 1.)
      f[i, j] = -2(2 - x[i]^2 - y[j]^2)
    elseif ipr == 2
      u_e[i, j] = (
        sin(2π * x[i]) * sin(2π * y[j]) +
        c1 * sin(16π * x[i]) * sin(16π * y[j])
      )
      f[i, j] = (
        4c2 * sin(2π * x[i]) * sin(2π * y[j]) +
        c2 * sin(16π * x[i]) * sin(16π * y[j])
      )
    end
    u_n[i, j] = 0.
  end end

  @views begin
    u_n[:, 1] = u_e[:, 1]
    u_n[:, ny+1] = u_e[:, ny+1]

    u_n[1, :] = u_e[1, :]
    u_n[nx+1, :] = u_e[nx+1, :]
  end

  r = zeros(Float64, nx + 1, ny + 1)

  for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
    write(initial, "$(x[i]) $(y[j]) $(f[i, j]) $(u_n[i, j]) $(u_e[i, j])\n")
  end end

  val, t, bytes, gctime, memallocs = @timed begin
    gauss_seidel(Δx, Δy, nx, ny, r, f, u_n, max_iter, tol, out)
  end

  u_error = u_n - u_e
  rms_error = compute_l2norm(nx, ny, u_error)
  max_error = maximum(abs.(u_error))

  write(out, "Error details:\n")
  write(out, "L-2 Norm=$rms_error\n")
  write(out, "Maximum Norm=$max_error\n")
  write(out, "CPU Time=$t\n")

  for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
    write(final, "$(x[i]) $(y[j]) $(f[i, j]) $(u_n[i, j]) $(u_e[i, j])\n")
  end end

  close(initial)
  close(final)
  close(out)
  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end