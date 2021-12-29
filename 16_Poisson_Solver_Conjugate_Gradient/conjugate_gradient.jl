include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Printf

conjugate_gradient(Δx, Δy, nx, ny, r, f, u_n, max_iter, tol, out, ε=1e-16, freq=100) = begin
  # create text file for writing residual history
  res = open("cg_residual.txt", "w")
  # write(res, "k"," ","rms"," ","rms/rms0"," \n")

  compute_residual(nx, ny, Δx, Δy, f, u_n, r)
  init_rms = rms = compute_l2norm(nx, ny, r)

  println("0 $rms $(rms / init_rms)")
  # allocate the matric for direction and set the initial direction (conjugate vector)
  p = zeros(Float64, nx + 1, ny + 1)
  ∇p = zero(p)  # same type & shape as p, filled with zeros

  # assign conjugate vector to initial residual
  @unroll p[1:nx+1, 1:ny+1] = r[1:nx+1, 1:ny+1]

  # start calculation
  it = 0
  while it < max_iter
    it += 1

    # calculate ∇^2(residual)
    @fastmath @simd for j ∈ 2:ny for i ∈ 2:nx
      ∇p[i, j] = (
        (p[i+1, j] - 2p[i, j] + p[i-1, j]) / Δx^2 +
        (p[i, j+1] - 2p[i, j] + p[i, j-1]) / Δy^2
      )
    end end

    aa = bb = 0.
    # calculate aa, bb, cc. cc is the distance parameter(α_n)
    @unroll begin
      aa += r[2:nx, 2:ny]^2
      bb += ∇p[2:nx, 2:ny] * p[2:nx, 2:ny]
    end
    # cc = <r,r>/<d,p>
    cc = aa / (bb + ε)

    # update the numerical solution by adding some component of conjugate vector
    @unroll u_n[2:nx, 2:ny] += cc * p[2:nx, 2:ny]

    # bb = <r,r> = aa (calculated in previous loop)
    bb = aa
    aa = 0.

    # update the residual by removing some component of previous residual
    @fastmath for j ∈ 2:ny for i ∈ 2:nx  # non-simd & non-unroll
      r[i, j] -= cc * ∇p[i, j]
      aa += r[i, j]^2
    end end
    # cc = <r-cd, r-cd>/<r,r>
    cc = aa / (bb + ε)

    # update the conjugate vector
    @unroll p[1:nx, 1:ny] = r[1:nx, 1:ny] + cc * p[1:nx, 1:ny]

    # compute the l2norm of residual
    rms = compute_l2norm(nx, ny, r)

    if mod(it, freq) == 0
      write(res, "$it $rms $(rms / init_rms)\n")
      println("$it $rms $(rms / init_rms)")
    end

    if (rms / init_rms) <= tol break end
  end

  write(out, "L-2 Norm=$rms\n")
  write(out, "Maximum Norm=$(maximum(abs.(r)))\n")
  write(out, "Iterations=$it\n")
  close(res)
  return
end

main() = begin
  nx, ny = 512, 512
  max_iter = 20 * 100_000
  tol = 1e-9
  ipr = 1

  # create output file for L2-norm
  out = open("output.txt", "w")
  write(out, "Residual details:\n")
  # create text file for initial and final field
  initial = open("field_initial.txt", "w")
  final = open("field_final.txt", "w")

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

  c1 = (1. / 16.)^2
  c2 = -2π^2

  @simd for i ∈ 1:nx + 1 for j ∈ 1:ny + 1
    if ipr == 1
      u_e[i, j] = (x[i]^2 - 1) * (y[j]^2 - 1)
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
    conjugate_gradient(Δx, Δy, nx, ny, r, f, u_n, max_iter, tol, out)
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