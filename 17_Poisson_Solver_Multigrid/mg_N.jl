include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils

mg_N(Δx, Δy, nx, ny, r, f, u_n, v1, v2, v3, max_iter, tol, out, n_level, freq=1) = begin
  res = open("mg_residual.txt", "w")

  # define 3D matrix for u_mg
  u_mg, f_mg, r_mg, p_mg = (Matrix{Float64}[] for _ ∈ 1:4)
  # fine mesh numerical solution and source temp at first level of 3D matrix
  push!(u_mg, u_n)
  push!(f_mg, f)
  push!(r_mg, r)
  push!(p_mg, zero(r))

  # compute initial residual
  compute_residual(nx, ny, Δx, Δy, f_mg[1], u_mg[1], r)

  # compute initial L-2 norm
  init_rms = rms = compute_l2norm(nx, ny, r)

  println("0 $rms $(rms / init_rms)")

  if nx < 2^n_level println("Number of levels exceeds the possible number.\n") end
  # allocate memory for grid size at different levels
  lnx = zeros(Int, n_level)
  lny = zero(lnx)
  lΔx = zeros(Float64, n_level)
  lΔy = zero(lΔx)

  # initialize the mesh details at fine level
  lnx[1], lny[1] = nx, ny
  lΔx[1], lΔy[1] = Δx, Δy

  # calculate mesh details for coarse levels and allocate matrices for
  # numerical solution and error restricted from upper level
  for l ∈ 2:n_level
    lnx[l] = lnx[l-1] ÷ 2
    lny[l] = lny[l-1] ÷ 2
    lΔx[l] = 2lΔx[l-1]
    lΔy[l] = 2lΔy[l-1]

    arr = zeros(Float64, lnx[l] + 1, lny[l] + 1)  # allocate array for storage at coarse levels
    push!(u_mg, arr)
    push!(f_mg, zero(arr))
    push!(r_mg, zero(arr))  # temporaty residual which is restricted to coarse mesh error
    push!(p_mg, zero(arr))
  end

  it = 0
  while it < max_iter
    it += 1
    # call relaxation on fine grid and compute the numerical solution for fixed number of iterations
    gauss_seidel_mg(lnx[1], lny[1], lΔx[1], lΔy[1], f_mg[1], u_mg[1], v1)

    # check for convergence only for finest grid, compute the residual and L2 norm
    compute_residual(lnx[1], lny[1], lΔx[1], lΔy[1], f_mg[1], u_mg[1], r)

    # compute the l2norm of residual
    rms = compute_l2norm(lnx[1], lny[1], r)

    if mod(it, freq) == 0
      # write results only for finest residual
      write(res, "$it $rms $(rms / init_rms)\n")
      println("$it $rms $(rms / init_rms)")
    end

    if rms / init_rms <= tol break end

    # from second level to coarsest level
    # full multigrid scheme
    for k ∈ 2:n_level
      if k == 2
        # for second level temporary residual is taken from fine mesh level
        tmp_res = r
      else
        # from third level onwards residual is computed for (k-1) level
        # which will be restricted to kth lvel error
        compute_residual(lnx[k-1], lny[k-1], lΔx[k-1], lΔy[k-1], f_mg[k-1], u_mg[k-1], r_mg[k-1])
        tmp_res = r_mg[k-1]
      end
      # restrict residual from (k-1)th level to kth level
      restriction(lnx[k-1], lny[k-1], lnx[k], lny[k], tmp_res, f_mg[k])

      # NOTE: using indexing in fill: must use @views !
      @views fill!(u_mg[k], 0.)  # set solution at kth level to zero

      # solve (∇^-λ^2)ϕ = ϵ on coarse grid (kth level)
      gauss_seidel_mg(lnx[k], lny[k], lΔx[k], lΔy[k], f_mg[k], u_mg[k], k < n_level ? v1 : v2)
    end

    for k ∈ n_level:-1:2
      # prolongate solution from (k)th level to (k-1)th level
      prolongation(lnx[k], lny[k], lnx[k-1], lny[k-1], u_mg[k], p_mg[k-1])

      # correct the solution on (k-1)th level
      @simd for j ∈ 2:lny[k-1] for i ∈ 2:lnx[k-1]
        u_mg[k-1][i, j] += p_mg[k-1][i, j]
      end end

      gauss_seidel_mg(lnx[k-1], lny[k-1], lΔx[k-1], lΔy[k-1], f_mg[k-1], u_mg[k-1], v3)
      # println(u_mg[k-1])
    end
  end
  u_n = u_mg[1]

  write(out, "L-2 Norm=$rms\n")
  write(out, "Maximum Norm=$(maximum(abs.(r)))\n")
  write(out, "Iterations=$it\n")
  close(res)
  return
end

main() = begin
  nx, ny = 512, 512
  max_iter = 100_000
  n_level = 9
  tol = 1e-9
  ipr = 1

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
  u_e, u_n = (similar(f) for _ ∈ 1:2)

  for i ∈ 1:nx + 1
    x[i] = x_l + Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = y_b + Δy * (i - 1)
  end

  c1 = (1. / 16)^2
  c2 = -2π^2

  for i ∈ 1:nx + 1 for j ∈ 1:ny + 1
    if ipr == 1
      u_e[i, j] = (x[i]^2 - 1) * (y[j]^2 - 1)
      f[i, j] = -2(2 - x[i]^2 - y[j]^2)
    elseif ipr == 2
      u_e[i, j] = sin(2π * x[i]) * sin(2π * y[j]) + c1 * sin(16π * x[i]) * sin(16π * y[j])
      f[i, j] = 4c2 * sin(2π * x[i]) * sin(2π * y[j]) + c2 * sin(16π * x[i]) * sin(16π * y[j])
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
  init_rms = rms = 0.

  open("field_initial.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(u_n[i, j]) $(u_e[i, j])\n")
    end end
  end
  val, t, bytes, gctime, memallocs = @timed begin
    mg_N(Δx, Δy, nx, ny, r, f, u_n, v1, v2, v3, max_iter, tol, out, n_level)
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