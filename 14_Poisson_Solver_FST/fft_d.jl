include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using FFTW

ps_fst(nx, ny, Δx, Δy, f) = begin
  u = Array{Complex{Float64}}(undef, nx - 1, ny - 1)
  data, data1, e = (similar(u) for _ ∈ 1:3)

  @unroll data[1:nx-1, 1:ny-1] = f[2:nx, 2:ny]

  e = FFTW.r2r(data, FFTW.RODFT00)

  @fastmath @simd for j ∈ 1:ny - 1 for i ∈ 1:nx - 1
    data1[i, j] = e[i, j] / (
      (2. / Δx^2) * (cos(π * i / nx) - 1.) +
      (2. / Δy^2) * (cos(π * j / ny) - 1.)
    )
  end end

  return FFTW.r2r(data1, FFTW.RODFT00) / (2nx * 2ny)
end

main() = begin
  nx, ny = 128, 128

  x_l, x_r = 0., 1.
  y_b, y_t = 0., 1.

  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  x = Array{Float64}(undef, nx + 1)
  y = similar(x)
  f = Array{Float64}(undef, nx + 1, ny + 1)
  ue = similar(f)
  un = similar(f)

  for i ∈ 1:nx + 1
    x[i] = x_l + Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = y_b + Δy * (i - 1)
  end

  # given exact solution
  km = 16
  c1 = (1. / km)^2
  c2 = -8π^2

  for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
    ue[i, j] = (
      sin(2π * x[i]) * sin(2π * y[j]) +
      c1 * sin(km * 2π * x[i]) * sin(km * 2π * y[j])
    )
    f[i, j] = (
      c2 * sin(2π * x[i]) * sin(2π * y[j]) +
      c2 * sin(km * 2π * x[i]) * sin(km * 2π * y[j])
    )
    un[i, j] = 0.
  end end

  open("field_initial.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(un[i, j]) $(ue[i, j])\n")
    end end
  end

  val, t, bytes, gctime, memallocs = @timed begin
    un[2:nx, 2:ny] = ps_fst(nx, ny, Δx, Δy, f)
  end

  uerror = un - ue
  rms_error = compute_l2norm_bnds(nx, ny, uerror)
  max_error = maximum(abs.(uerror))

  open("output.txt", "w") do io
    write(io, "Error details:\n")
    write(io, "L-2 Norm=$rms_error\n")
    write(io, "Maximum Norm=$max_error\n")
    write(io, "CPU Time=$t\n")
  end

  open("field_final.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(un[i, j]) $(ue[i, j])\n")
    end end
  end

  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end