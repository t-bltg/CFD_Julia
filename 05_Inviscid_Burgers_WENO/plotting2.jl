using DelimitedFiles
using PyPlot
using CSV

rc("font", size=16.)
ns = 10
tf = .25
nx = 200

solution = readdlm("solution_$nx.txt")
x = solution[:, 1]
u = solution[:, 2:ns+1]

fig2 = figure("An example", figsize=(8, 6))
ax3 = fig2[:add_subplot](1, 1, 1)

for i ∈ 1:ns - 1
  ax3.plot(x, u[:, i], lw=1, label="t=$(tf*i/ns)")
end
ax3.set_xlabel("\$x\$")
ax3.set_ylabel("\$u\$")
ax3.set_xlim(0, 1)
ax3.legend(fontsize=14, loc=0, bbox_to_anchor=(.3, .45, .5, .5))

fig2.tight_layout()
fig2.savefig("burgers_cds.pdf")
