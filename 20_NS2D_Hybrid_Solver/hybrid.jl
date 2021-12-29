include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils
using Plots
using FFTW

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 2nd-order finite difference discretization
# -----------------------------------------------------------------------------#
numerical(nx, ny, nt, Δx, Δy, Δt, re, x, y, wn, ns) = begin
  data = Array{Complex{Float64}}(undef, nx, ny)
  w₁f, w₂f, wₙf, j₁f, j₂f, jₙf = (similar(data) for _ ∈ 1:6)

  ut = Array{Float64}(undef, nx + 1, ny + 1)
  d₁, d₂, d₃ = (similar(data) for _ ∈ 1:3)

  m = 1  # record index
  freq = nt ÷ ns

  @unroll data[1:nx, 1:ny] = complex(wn[2:nx+1, 2:ny+1], 0.)

  k₂ = wavespace(nx, ny, Δx, Δy)
  wₙf = fft(data)
  wₙf[1, 1] = 0.

  α₁, α₂, α₃ = 8. / 15., 2. / 15., 1. / 3.
  γ₁, γ₂, γ₃ = 8. / 15., 5. / 12., 3. / 4.
  ρ₂, ρ₃ = -17. / 60., -5. / 12.

  @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
    z = .5Δt * k₂[i, j] / re
    d₁[i, j] = α₁ * z
    d₂[i, j] = α₂ * z
    d₃[i, j] = α₃ * z
  end end

  for k ∈ 1:nt
    jₙf = jacobian(nx, ny, Δx, Δy, wₙf, k₂)

    @unroll w₁f[1:nx, 1:ny] = (
      ((1. - d₁[1:nx, 1:ny]) / (1. + d₁[1:nx, 1:ny])) * wₙf[1:nx, 1:ny] +
      (γ₁ * Δt * jₙf[1:nx, 1:ny]) / (1. + d₁[1:nx, 1:ny])
    )

    w₁f[1, 1] = 0.
    j₁f = jacobian(nx, ny, Δx, Δy, w₁f, k₂)

    @unroll w₂f[1:nx, 1:ny] = (
      ((1. - d₂[1:nx, 1:ny]) / (1. + d₂[1:nx, 1:ny])) * w₁f[1:nx, 1:ny] + (
        ρ₂ * Δt * jₙf[1:nx, 1:ny] +
        γ₂ * Δt * j₁f[1:nx, 1:ny]
      ) / (1. + d₂[1:nx, 1:ny])
    )

    w₂f[1, 1] = 0.
    j₂f = jacobian(nx, ny, Δx, Δy, w₂f, k₂)

    @unroll wₙf[1:nx, 1:ny] = (
      ((1. - d₃[1:nx, 1:ny]) / (1. + d₃[1:nx, 1:ny])) * w₂f[1:nx, 1:ny] + (
        ρ₃ * Δt * j₁f[1:nx, 1:ny] +
        γ₃ * Δt * j₂f[1:nx, 1:ny]
      ) / (1. + d₃[1:nx, 1:ny])
    )

    if mod(k, freq) == 0
      @show k
      ut[1:nx, 1:ny] = real(ifft(wₙf))
      # periodic BC
      @views begin
        ut[nx+1, :] = ut[1, :]
        ut[:, ny+1] = ut[:, 1]
      end
      open("vm$m.txt", "w") do io
        for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
          write(io, "$(x[i]) $(y[j]) $(ut[i, j])\n")
        end end
        m += 1
      end
    end
  end
  return ut
end

# -----------------------------------------------------------------------------#
# Calculate Jacobian in fourier space
# jf = -J(w,ψ)
# -----------------------------------------------------------------------------#
jacobian(nx, ny, Δx, Δy, wf, k2) = begin
  w = Array{Float64}(undef, nx + 2, ny + 2)
  s = similar(w)

  data = Array{Complex{Float64}}(undef, nx, ny)
  sf, jf = (similar(data) for _ ∈ 1:2)

  # Arakawa numerical scheme for Jacobian
  gg = 1. / (4. * Δx * Δy)
  hh = 1. / 3.

  w[2:nx+1, 2:ny+1] = real(ifft(wf))

  @views begin
    # periodic BC
    w[nx+2, :] = w[2, :]
    w[:, ny+2] = w[:, 2]

    # ghost points
    w[1, :] = w[nx+1, :]
    w[:, 1] = w[:, ny+1]
  end

  @unroll sf[1:nx, 1:ny] = wf[1:nx, 1:ny] / k2[1:nx, 1:ny]

  s[2:nx+1, 2:ny+1] = real(ifft(sf))

  @views begin
    # periodic BC
    s[nx+2, :] = s[2, :]
    s[:, ny+2] = s[:, 2]

    # ghost points
    s[1, :] = s[nx+1, :]
    s[:, 1] = s[:, ny+1]
  end

  @fastmath @simd for j ∈ 2:ny + 1 for i ∈ 2:nx + 1
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

    data[i-1, j-1] = complex(-(j1 + j2 + j3) * hh, 0.)
  end end
  return fft(data)
end

main() = begin
  nx, ny = 128, 128

  x_l, x_r = 0., 2π
  y_b, y_t = 0., 2π

  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  Δt, tf = .01, 20.
  nt = Int(tf / Δt)
  re = 1000.
  ns = 10

  x = Array{Float64}(undef, nx + 1)
  y = Array{Float64}(undef, ny + 1)

  for i ∈ 1:nx + 1
    x[i] = Δx * (i - 1)
  end
  for i ∈ 1:ny + 1
    y[i] = Δy * (i - 1)
  end

  wn = Array{Float64}(undef, nx + 2, ny + 2)
  vm_ic(nx, ny, x, y, wn)

  @views begin
    # ghost points
    wn[1, :] = wn[nx+1, :]
    wn[:, 1] = wn[:, ny+1]

    wn[nx+2, :] = wn[2, :]
    wn[:, ny+2] = wn[:, 2]
  end

  un0 = wn[2:nx+2, 2:ny+2]

  open("vm0.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j]) $(un0[i, j])\n")
    end end
  end

  un = numerical(nx, ny, nt, Δx, Δy, Δt, re, x, y, wn, ns)

  open("field_final.txt", "w") do io
    for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
      write(io, "$(x[i]) $(y[j])  $(un[i, j])\n")
    end end
  end

  p1 = contour(x, y, transpose(un), fill=true, xlabel="\$X\$", ylabel="\$Y\$", title="Numerical")
  savefig(p1, "vm.pdf")
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end