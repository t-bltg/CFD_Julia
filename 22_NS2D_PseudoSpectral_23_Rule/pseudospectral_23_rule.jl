include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils
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

  wₙf = fft(data)
  k₂ = wavespace(nx, ny, Δx, Δy)
  wₙf[1, 1] = 0

  α₁, α₂, α₃ = 8 / 15, 2 / 15, 1 / 3
  γ₁, γ₂, γ₃ = 8 / 15, 5 / 12, 3 / 4
  ρ₂, ρ₃ = -17 / 60, -5 / 12

  @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
    z = .5Δt * k₂[i, j] / re
    d₁[i, j] = α₁ * z
    d₂[i, j] = α₂ * z
    d₃[i, j] = α₃ * z
  end end

  for k ∈ 1:nt
    jₙf = jacobian(nx, ny, Δx, Δy, wₙf, k₂)

    # 1st step
    @unroll w₁f[1:nx, 1:ny] = (
      ((1 - d₁[1:nx, 1:ny]) / (1 + d₁[1:nx, 1:ny])) * wₙf[1:nx, 1:ny] +
      (γ₁ * Δt * jₙf[1:nx, 1:ny]) / (1 + d₁[1:nx, 1:ny])
    )

    w₁f[1, 1] = 0
    j₁f = jacobian(nx, ny, Δx, Δy, w₁f, k₂)

    # 2nd step
    @unroll w₂f[1:nx, 1:ny] = (
      ((1 - d₂[1:nx, 1:ny]) / (1 + d₂[1:nx, 1:ny])) * w₁f[1:nx, 1:ny] + (
        ρ₂ * Δt * jₙf[1:nx, 1:ny] +
        γ₂ * Δt * j₁f[1:nx, 1:ny]
      ) / (1 + d₂[1:nx, 1:ny])
    )

    w₂f[1, 1] = 0
    j₂f = jacobian(nx, ny, Δx, Δy, w₂f, k₂)

    # 3rd step
    @unroll wₙf[1:nx, 1:ny] = (
      ((1 - d₃[1:nx, 1:ny]) / (1 + d₃[1:nx, 1:ny])) * w₂f[1:nx, 1:ny] + (
        ρ₃ * Δt * j₁f[1:nx, 1:ny] +
        γ₃ * Δt * j₂f[1:nx, 1:ny]
      ) / (1 + d₃[1:nx, 1:ny])
    )

    if mod(k, freq) == 0
      @show k
      ut[1:nx, 1:ny] = real(ifft(wₙf))
      @views begin
        # periodic BC
        ut[nx+1, :] = ut[1, :]
        ut[:, ny+1] = ut[:, 1]
      end
      open("vm$m.txt", "w") do io
        for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
          write(io, "$(x[i]) $(y[j]) $(ut[i, j])\n")
        end end
      end
      m += 1
    end
  end

  return ut
end

# -----------------------------------------------------------------------------#
# Calculate Jacobian in fourier space
# jf = -J(w,ψ)
# -----------------------------------------------------------------------------#
jacobian(nx, ny, Δx, Δy, wf, k₂, ε=1e-6) = begin
  kx = Array{Float64}(undef, nx)
  ky = Array{Float64}(undef, ny)

  # wave number indexing
  hx = 2π / (nx * Δx)

  @fastmath @simd for i ∈ 1:nx÷2
    kx[i] = hx * (i - 1.)
    kx[i+nx÷2] = hx * (i - nx ÷ 2 - 1)
  end
  kx[1] = ε
  ky = transpose(kx)

  j₁f = zeros(ComplexF64, nx, ny)
  j₂f, j₃f, j₄f = (zero(j₁f) for _ ∈ 1:3)

  # x-derivative
  @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
    j₁f[i, j] = 1im * wf[i, j] * kx[i] / k₂[i, j]
    j₄f[i, j] = 1im * wf[i, j] * kx[i]
  end end

  # y-derivative
  @fastmath @simd for i ∈ 1:nx for j ∈ 1:ny
    j₂f[i, j] = 1im * wf[i, j] * ky[j]
    j₃f[i, j] = 1im * wf[i, j] * ky[j] / k₂[i, j]
  end end

  nxe = Int(floor(2nx / 3))
  nye = Int(floor(2ny / 3))

  @fastmath @simd for j ∈ 1:ny for i ∈ Int(floor(nxe / 2) + 1):Int(nx - floor(nxe / 2))
    j₁f[i, j] = j₂f[i, j] = j₃f[i, j] = j₄f[i, j] = 0.
  end end

  @fastmath @simd for j ∈ Int(floor(nye / 2) + 1):Int(ny - floor(nye / 2)) for i ∈ 1:nx
    j₁f[i, j] = j₂f[i, j] = j₃f[i, j] = j₄f[i, j] = 0.
  end end

  j₁, j₂, j₃, j₄ = real(ifft(j₁f)), real(ifft(j₂f)), real(ifft(j₃f)), real(ifft(j₄f))
  jacp = zeros(Float64, nx, ny)

  @unroll jacp[1:nx, 1:ny] = (
    j₁[1:nx, 1:ny] * j₂[1:nx, 1:ny] -
    j₃[1:nx, 1:ny] * j₄[1:nx, 1:ny]
  )

  return fft(jacp)
end

main() = begin
  nx, ny = 128, 128

  x_l, x_r = 0., 2π
  y_b, y_t = 0., 2π

  Δx = (x_r - x_l) / nx
  Δy = (y_t - y_b) / ny

  Δt, tf = .01, 20.
  @show nt = Int(tf / Δt)
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
      write(io, "$(x[i]) $(y[j]) $(un[i, j])\n")
    end end
  end
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end