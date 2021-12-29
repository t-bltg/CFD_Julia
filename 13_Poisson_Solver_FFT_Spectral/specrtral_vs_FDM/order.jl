using DelimitedFiles
using PyPlot
using CSV

rc("font", size=16.)

grid = [32, 64, 128, 256, 512]

if false
  fft_spectral = [1.339154672220572e-16, 1.342489800590083e-16, 1.3269947532677886e-16, 1.4514549677756118e-16, 1.485803753784319e-16]
  fft_fdm = [.0015607100315532957, .0005987381110678801, .00014313734718665358, 3.549617203207291e-5, 8.865373334924762e-6]
  xr² = [64, 256]
  yr² = [2e-3, 1.25e-4]
  xr¹ = yr¹ = nothing
else
  fft_fdm, fft_spectral = [], []
  err_l2(fn) = parse(float(Int), split(readlines(fn)[2], "=")[2])
  for n ∈ grid
    push!(fft_spectral, err_l2("../output_$n.txt"))
    push!(fft_fdm, err_l2("../../12_Poisson_Solver_FFT/output_$n.txt"))
  end
  # y = k⋅x^s ⟹ log(y) = log(k) + s⋅log(x) ⟹ Y = s⋅X + b, with s the slope and b the intercept.
  # given two points (x₁, y₁) and (x₂, y₂) in the loglog plot:
  # s = log(y₂) - log(y₁) / (log(x₂) - log(x₁)) = log(y₂ / y₁) / log(x₂ / x₁)
  # ⟹ y₂ = y₁⋅exp(s⋅log(x₂ / x₁))
  y₂(s, x₁, x₂, y₁) = y₁ * exp(s * log(x₂ / x₁))
  s² = -2  # second order slope
  xr² = [64, 256]
  yr² = [2e-3, Inf]
  @show yr²[end] = y₂(s², xr²[begin], xr²[end], yr²[begin])
  s¹ = -1  # first order slope
  xr¹ = xr²
  yr¹ = [2e-3, Inf]
  @show yr¹[end] = y₂(s¹, xr¹[begin], xr¹[end], yr¹[begin])
end

fig = figure("An example", figsize=(14, 6))
ax1 = fig[:add_subplot](1, 2, 1)
ax2 = fig[:add_subplot](1, 2, 2)

ax1.plot(grid, fft_spectral, color="red", lw=4, marker="o", markeredgecolor="k", markersize=12)
ax1.set_xscale("log", base=2)
ax1.set_yscale("log", base=10)
ax1.set_ylim([1e-16, .2e-15])
ax1.set_xlabel("\$N\$")
ax1.set_ylabel("\$|ϵ|_2\$")
ax1.set_title("Spectral method")

ax2.plot(grid, fft_fdm, color="blue", lw=4, marker="o", markeredgecolor="k", markersize=12)

if !isnothing(xr¹)
  ax2.plot(xr¹, yr¹, color="black", lw=2, ls="-.")
  ax2.text(xr¹[end] + 15, yr¹[end], "Slope=-1")
end
ax2.plot(xr², yr², color="black", lw=2, ls="--")
ax2.text(xr²[end] + 15, yr²[end], "Slope=-2")

ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=10)
ax2.set_ylim([4e-6, 4e-3])
ax2.set_xlabel("\$N\$")
ax2.set_ylabel("\$|ϵ|_2\$")
ax2.set_title("Second-order CDS")

fig.tight_layout()
fig.savefig("order_FFT.pdf")
