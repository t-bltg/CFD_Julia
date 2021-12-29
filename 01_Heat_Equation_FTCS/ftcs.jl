# clearconsole()
include("../Common.jl")  # weird, but seems to work
using .Common  # local import
using BenchmarkTools
using CPUTime
using Unroll
using Plots

main() = begin
  x_l, x_r = -1., 1.
  Δx = .025
  nx = Int((x_r - x_l) / Δx)

  t, Δt = 1., .0025
  nt = Int(t / Δt)

  α = 1. / π^2

  x = Array{Float64}(undef, nx + 1)
  u_e = similar(x)
  un = Array{Float64}(undef, nx + 1, nt + 1)

  for i ∈ 1:nx + 1
    x[i] = x_l + Δx * (i - 1)  # location of each grid point
    un[i, begin] = -sin(π * x[i])  # initial condition @ t=0
    u_e[i] = -exp(-t) * sin(π * x[i])  # initial condition @ t=0
  end

  un[begin, begin] = 0.
  un[end, begin] = 0.

  beta = α * Δt / Δx^2

  @time (
    for k ∈ 2:nt + 1
      # @unroll is safe since we pick old values (k-1) to update new values (k)
      @unroll un[2:nx, k]=un[2:nx, k-1] + beta * (un[3:nx+1, k-1] - 2un[2:nx, k-1] + un[1:nx-1, k-1])
      un[begin, k]=0.  # boundary condition at x=-1
      un[end, k]=0.  # boundary condition at x=-1
    end
  )

  # compute L2 norm of the error
  u_error = un[:, nt+1] - u_e
  rms_error = compute_l2norm(nx, u_error)

  # create output file for L2-norm
  open("output.txt", "w") do f
    write(f, "Error details:\n")
    write(f, "L-2 Norm=$rms_error\n")
    write(f, "Maximum Norm=$(maximum(abs.(u_error)))\n")
  end

  # create text file for final field
  open("field_final.csv", "w") do f
    write(f, "x ue un uerror\n")
    for i ∈ 1:nx + 1
      write(f, "$(x[i]) $(u_e[i]) $(un[i, nt+1]) $(u_error[i])\n")
    end
  end

  run(`cat output.txt`)
  return
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end