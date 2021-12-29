using DelimitedFiles
using PyPlot
using CSV

rc("font", size=16.)  # family="Arial",
ns = 10
tf = .25
nx = 200

solution = readdlm("solution_$nx.txt")
x = solution[:, 1]
u = solution[:, 2:ns+1]

solution = readdlm("solution_d_$nx.txt")
xd = solution[:, 1]
ud = solution[:, 2:ns+1]

solution = readdlm("solution_p_$nx.txt")
xp = solution[:, 1]
up = solution[:, 2:ns+1]

fig = figure("An example", figsize=(14, 6))
ax1 = fig[:add_subplot](1, 2, 1)
ax2 = fig[:add_subplot](1, 2, 2)

for i ∈ 1:ns
  ax1.plot(xd, ud[:, i], lw=1, label="t=$(tf*i/ns)")
end
ax1.set_xlabel("\$x\$")
ax1.set_ylabel("\$u\$")
ax1.set_title("Dirichlet boundary")
ax1.set_xlim(0, 1)
ax1.legend(fontsize=14, loc=0, bbox_to_anchor=(.55, .35, .5, .5))

for i ∈ 1:ns
  ax2.plot(xp, up[:, i], lw=1, label="t=$(tf*i/ns)")
end

ax2.set_xlabel("\$x\$")
ax2.set_ylabel("\$u\$")
ax2.set_title("Periodic boundary")
ax2.set_xlim(0, 1)
ax2.legend(fontsize=14, loc=0, bbox_to_anchor=(.55, .35, .5, .5))

# plt[:subplot](ax1)
# plt[:subplot](ax2)

fig.tight_layout()
fig.savefig("weno_2.pdf")
