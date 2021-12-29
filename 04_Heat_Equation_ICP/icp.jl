include("../Common.jl")
using .Common
using BenchmarkTools
using Unroll
using Printf
using Utils

numerical(nx, nt, dx, dt, x, u_n, α) = begin
  a, b, c, r = (similar(x) for _ ∈ 1:4)

  for k ∈ 2:nt + 1
    uv = view(u_n, :, k - 1)
    uv[begin], uv[end] = 0., 0.

    @unroll begin
      r[2:nx] = (
        - 2. / (α * dt) * (uv[3:nx+1] + 10uv[2:nx] + uv[1:nx-1])
        - 12. / dx^2 * (uv[3:nx+1] - 2uv[2:nx] + uv[1:nx-1])
      )
      a[2:nx] = +12 / dx^2 - 2 / (α * dt)
      b[2:nx] = -24 / dx^2 - 20 / (α * dt)
      c[2:nx] = +12 / dx^2 - 2 / (α * dt)
    end
    a[begin], b[begin], c[begin], r[begin] = 0., 1., 0., 0.
    a[end], b[end], c[end], r[end] = 0., 1., 0., 0.

    tdma(a, b, c, r, view(u_n, :, k), 1, nx)
  end
end

main() = begin
  x_l, x_r = -1., 1.
  dx = .025
  nx = Int((x_r - x_l) / dx)

  t, dt = 1., .0025
  nt = Int(t / dt)

  α = 1. / π^2

  x = Array{Float64}(undef, nx + 1)
  tlist = Array{Float64}(undef, nt + 1)
  u_n = Array{Float64}(undef, nx + 1, nt + 1)
  u_e = similar(x)

  for i ∈ 1:nx + 1
    x[i] = x_l + dx * (i - 1)  # location of each grid point
    u_n[i, begin] = -sin(π * x[i])  # initial condition @ t=0
    u_e[i] = -exp(-t) * sin(π * x[i])  # initial condition @ t=0
  end

  if boolenv("BENCH")
    @btime numerical($nx, $nt, $dx, $dt, $x, $u_n, $α)
  else
    @time numerical(nx, nt, dx, dt, x, u_n, α)
  end

  u_error = u_n[:, nt+1] - u_e
  rms_error = compute_l2norm(nx, u_error)

  # create output file for L2-norm
  open("output.txt", "w") do f
    write(f, "Error details:\n")
    write(f, "L-2 Norm=$(rms_error)\n")
    write(f, "Maximum Norm=$(maximum(abs.(u_error)))\n")
  end

  # create text file for final field
  open("field_final.csv", "w") do f
    write(f, "x ue un uerror\n")
    for i ∈ 1:nx + 1
      write(f, "$(x[i]) $(u_e[i]) $(u_n[i, nt+1]) $(u_error[i])\n")
    end
  end
  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end