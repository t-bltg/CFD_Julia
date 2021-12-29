include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Printf
using FFTW

ps_spectral(nx, ny, Δx, Δy, f, ε=1e-6) = begin
  kx = Array{Float64}(undef, nx)
  ky = Array{Float64}(undef, ny)

  u = Array{Complex{Float64}}(undef, nx, ny)
  data, data1, e = (similar(u) for _ ∈ 1:3)

  # apply the FFT on the analysis on the continuous poisson equation:
  # ∂²u / ∂x² + ∂²u / ∂y² = f
  # ⟹ u^~_{m,n} = f^~_{m,n} / (m² + n²)
  hx = 2π / (nx * Δx)  # wave number indexing
  # hy = 2π / (ny * Δy)

  @fastmath @simd for i ∈ 1:nx÷2
    kx[i] = hx * (i - 1.)
    kx[i+nx÷2] = hx * (i - nx ÷ 2 - 1.)
  end
  kx[1] = ε
  ky = kx

  @unroll data[1:nx, 1:ny] = complex(f[1:nx, 1:ny], 0.)

  e = fft(data)
  e[1, 1] = 0.
  @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
    data1[i, j] = e[i, j] / (-kx[i]^2 - ky[j]^2)
  end end

  return real(ifft(data1))
end

main() = begin
  x_l, x_r = 0., 1.
  y_b, y_t = 0., 1.

  for nx ∈ (32, 64, 128, 256, 512)
    ny = nx

    Δx = (x_r - x_l) / nx
    Δy = (y_t - y_b) / ny

    x = Array{Float64}(undef, nx + 1)
    y = Array{Float64}(undef, ny + 1)
    f = Array{Float64}(undef, nx + 1, ny + 1)
    ue = similar(f)
    un = similar(f)

    for i ∈ 1:nx + 1
      x[i] = x_l + Δx * (i - 1)
    end
    for i ∈ 1:ny + 1
      y[i] = y_b + Δy * (i - 1)
    end

    # given exact solution (mms)
    km = 16
    c1 = (1. / km)^2
    c2 = -8π^2

    @simd for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
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

    open("field_initial_$nx.txt", "w") do io
      for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
        write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(un[i, j]) $(ue[i, j])\n")
      end end
    end

    val, t, bytes, gctime, memallocs = @timed begin
      un[1:nx, 1:ny] = ps_spectral(nx, ny, Δx, Δy, f)
    end

    # Periodic boundary condition
    @views begin
      un[nx+1, :] = un[1, :]
      un[:, ny+1] = un[:, 1]
    end

    uerror = un - ue
    rms_error = compute_l2norm_bnds(nx, ny, uerror)
    max_error = maximum(abs.(uerror))

    open("output_$nx.txt", "w") do io
      write(io, "Error details:\n")
      write(io, "L-2 Norm=$rms_error\n")
      write(io, "Maximum Norm=$max_error\n")
      write(io, "CPU Time=$t\n")
    end

    open("field_final_$nx.txt", "w") do io
      for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
        write(io, "$(x[i]) $(y[j]) $(f[i, j]) $(un[i, j]) $(ue[i, j])\n")
      end end
    end
    run(`cat output_$nx.txt`)
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end