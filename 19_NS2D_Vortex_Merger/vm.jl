include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Utils

# -----------------------------------------------------------------------------#
# Compute numerical solution
#   - Time integration using Runge-Kutta third order
#   - 2nd-order finite difference discretization
# -----------------------------------------------------------------------------#
numerical(nx, ny, nt, Δx, Δy, Δt, re, x, y, wn, ns) = begin
  wt = Array{Float64}(undef, nx + 2, ny + 2)  # temporary array during RK3 integration
  r, s = (similar(wt) for _ ∈ 1:2)

  f = Array{Float64}(undef, nx, ny)

  u = Array{Complex{Float64}}(undef, nx, ny)
  e, data, data1 = (similar(u) for _ ∈ 1:3)

  m = 1  # record index
  freq = nt ÷ ns

  for k ∈ 1:nt
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

    if mod(k, freq) == 0
      @show k
      ut = wn[2:nx+2, 2:ny+2]
      open("vm$m.txt", "w") do io
        for j ∈ 1:ny + 1 for i ∈ 1:nx + 1
          write(io, "$(x[i]) $(y[j]) $(ut[i, j])\n")
        end end
      end
    end
  end

  return wn[2:nx+2, 2:ny+2]
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

  val, t, bytes, gctime, memallocs = @timed begin
    un = numerical(nx, ny, nt, Δx, Δy, Δt, re, x, y, wn, ns)
  end

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
