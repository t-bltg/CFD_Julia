using DelimitedFiles
using PyPlot
using CSV

rc("font", size=16.)

gs_residual = readdlm("../15_Poisson_Solver_Gauss_Seidel/gs_residual.txt")
gs_iter_hist = gs_residual[:, 1]
gs_res_hist = convert(Matrix, gs_residual[:, 2:3])

cg_residual = readdlm("../16_Poisson_Solver_Conjugate_Gradient/cg_residual.txt")
cg_iter_hist = cg_residual[:, 1]
cg_res_hist = convert(Matrix, cg_residual[:, 2:3])

mg_residual = readdlm("mg_residual.txt")
mg_iter_hist = mg_residual[:, 1]
mg_res_hist = convert(Matrix, mg_residual[:, 2:3])

fig = figure("An example", figsize=(14, 12))
ax1 = fig[:add_subplot](2, 2, 1)
ax2 = fig[:add_subplot](2, 2, 2)
ax3 = fig[:add_subplot](2, 2, 3)
ax4 = fig[:add_subplot](2, 2, 4)

ax1.semilogy(gs_iter_hist, gs_res_hist[:, 2], color="blue", lw=4, label="Gauss-Seidel method")
ax1.set_xlabel("Iteration count")
ax1.set_ylabel("\$|r|_2\$")
ax1.legend()

ax2.semilogy(cg_iter_hist, cg_res_hist[:, 2], color="orange", lw=4, label="Conjugate-Gradient method")
ax2.set_xlabel("Iteration count")
ax2.set_ylabel("\$|r|_2\$")
ax2.legend()


ax3.semilogy(mg_iter_hist, mg_res_hist[:, 2], color="green", lw=4, label="Multigrid framework")
ax3.set_xlabel("Iteration count")
ax3.set_ylabel("\$|r|_2\$")
ax3.legend()

ax4.semilogy(gs_iter_hist, gs_res_hist[:, 2], color="blue", lw=4, label="Gauss-Seidel method")
ax4.semilogy(cg_iter_hist, cg_res_hist[:, 2], color="orange", lw=4, label="Conjugate-Gradient method")
ax4.semilogy(mg_iter_hist, mg_res_hist[:, 2], color="green", lw=4, label="Multigrid framework")
ax4.set_xlim(0, 2000)
ax4.set_xlabel("Iteration count")
ax4.set_ylabel("\$|r|_2\$")
ax4.legend()

fig.tight_layout()
fig.savefig("residual_poisson.pdf")

if false
  fig1 = figure("res1", figsize=(14, 6))
  ax3 = fig1[:add_subplot](1, 1, 1)

  ax3.semilogy(gs_iter_hist, gs_res_hist[:, 2], color="blue", lw=4, label="Gauss-Seidel method")

  ax3.set_xlabel("Iteration count")
  ax3.set_ylabel("\$|r|_2\$")
  ax3.legend()

  fig1.tight_layout()
  fig1.savefig("residual1.pdf")
end
