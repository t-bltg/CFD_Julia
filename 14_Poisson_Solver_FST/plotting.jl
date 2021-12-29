using DelimitedFiles
using PyPlot
using CSV

# rc("font", family="Times New Roman", size=18.0)
rc("font", size=16.)

nx = 128
ny = 128

init_field = readdlm("field_initial.txt")  # , type=Float64)
final_field = readdlm("field_final.txt")  # , datarow = 3, type=Float64)

x = convert(Array, init_field[:, 1])
y = convert(Array, init_field[:, 2])

u_e = convert(Array, init_field[:, 5])
u_e = reshape(u_e, (nx + 1, ny + 1))

u_n = convert(Array, final_field[:, 4])
u_n = reshape(u_n, (nx + 1, ny + 1))

xx = x[1:nx+1]
yy = reshape(y, (nx + 1, ny + 1))[1, :]

XX = repeat(xx, 1, length(yy))
XX = convert(Matrix, transpose(XX))
YY = repeat(yy, 1, length(xx))

fig = figure("An example", figsize=(14, 6))
ax1 = fig[:add_subplot](1, 2, 1)
ax2 = fig[:add_subplot](1, 2, 2)

cs1 = ax1.contourf(xx, yy, transpose(u_e), levels=20, cmap="jet", vmin=-1, vmax=1)
ax1.set_title("Exact solution")
# plt[:subplot](ax1); cs1

cs2 = ax2.contourf(xx, yy, transpose(u_n), levels=20, cmap="jet", vmin=-1, vmax=1)
ax2.set_title("Numerical solution")
# plt[:subplot](ax2); cs2

fig.colorbar(cs1, ax=ax1)
fig.colorbar(cs2, ax=ax2)

fig.tight_layout()
fig.savefig("fst_contour.pdf")
