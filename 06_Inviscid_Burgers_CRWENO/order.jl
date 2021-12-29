using Base.Broadcast
using DelimitedFiles
using Interpolations
using LinearAlgebra
using Plots; gr()
using Utils

@static if false
  using Surrogates  # slow to load (cuda)

  "for unstructured data"
  interp_unstructured(x, y, new_x) = begin
    mini, maxi = extrema(x)
    if !isscalar(mini) mini = collect(mini) end  # tuple to Array
    if !isscalar(maxi) maxi = collect(maxi) end
    itp = RadialBasis(x, y, mini, maxi)
    return itp.(new_x)
  end
end

"for grided data, faster"
interp_grid(x, y, new_x) = begin
  itp = interpolate((x,), y, Gridded(Linear()))
  return itp(new_x)
end

rectangle(x, y, w, h) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])

"origin + 2 vectors := triangle"
# triangle(o, v¹, v²) = Shape(o[1] .+ [0, v¹[1], v²[1]], o[2] .+ [0, v¹[2], v²[2]])
"three points"
triangle(points...) = Shape(collect(getfield.(points, :x)), collect(getfield.(points, :y)))

"given two points (x₁, y₁) and (x₂, y₂) and a slope s in the loglog plot, compute y₂"
y₂(s, x₁, x₂, y₁) = y₁ * exp(s * log(x₂ / x₁))

main() = begin
  ns = 6  # NOTE: singularities at the end (shock), avoid n > 6

  interp = true ? interp_grid : interp_unstructured

  per = Dict()
  dir = Dict()

  for (i, root) ∈ enumerate((".", "../05_Inviscid_Burgers_WENO"))
    for nx ∈ (100, 200, 400, 800, 1600)
      sol = readdlm("solution_d_$nx.txt")
      dir[nx] = Dict("x"=>sol[:, 1], "u"=>sol[:, ns])  # 2:ns+1
      sol = readdlm("solution_p_$nx.txt")
      per[nx] = Dict("x"=>sol[:, 1], "u"=>sol[:, ns])
    end

    """
    error behaves like |u@Δx - u| <= c.Δxᵖ, with c an arbitrary constant and Δx the mesh size

    ratio = (u@Δx - u@α⋅Δx) / (u@α⋅Δx - u@α²⋅Δx) = α⁻ᵖ, with α < 1.
    ⟹ p = - log(ratio) / log(α)

    or written alternatively
    ratio = (u@Δx - u@Δx/β) / (u@Δx/β - u@Δx/β²) = βᵖ, with β > 1.
    ⟹ p = log(ratio) / log(β)

    if we use N = f(1. / Δx) (number of nodes function of the inverse of the mesh size Δx)
    the sign of p is reversed
    """

    coarse, mid, fine = 200, 400, 800
    @assert (beta=mid / coarse) > 1
    x = dir[coarse]["x"]
    u = interp(dir[mid]["x"], dir[mid]["u"], x)
    for ord ∈ (1, 2, Inf)
      @show e¹ = norm(dir[coarse]["u"] - u, ord)  # Δx - Δx / 2
      @show e² = norm(u - interp(dir[fine]["x"], dir[fine]["u"], x), ord)  # Δx / 2 - Δx / 4
      @show p = log(e¹ / e²) / log(beta)
      @show s = Int(round(-p))  # NOTE: we use N instead of Δx, must negate the slope
      @show ord

      if ord == 2
        if false
          scatter(x, u, ms=2)
          plot!(dir[coarse]["x"], dir[coarse]["u"])
          savefig("sol-$i.pdf")
        end

        # FIXME: only :log10 is supported
        δx, δy = .25abs(mid - coarse), .25abs(e² - e¹)
        o = (x=.5(coarse + mid), y=.5(e¹ + e²))  # origin
        p₁ = map(+, o, (x=δx, y=0.))
        p₂ = (x=p₁.x, y=y₂(s, o.x, p₁.x, o.y))
        plot(triangle(o, p₁, p₂), opacity=.25, color=:gray, legend=false)
        plot!(
          [coarse, mid], [e¹, e²],
          xlabel="Δx", ylabel="ε", yaxis=:log10, yguidefontrotation=-90.,
          minorticks=true, minorgrid=true
        )
        annotate!(map(+, o, (x=0., y=.5δy))..., text("slope=$(round(s, digits=2))", 10))
        savefig("order-$i.pdf")
      end
    end

    coarse, mid, ref = 200, 400, 1600
    @assert (beta=mid / coarse) > 1
    x = dir[coarse]["x"]
    ue = interp(dir[ref]["x"], dir[ref]["u"], x)  # quasi exact
    for ord ∈ (1, 2, Inf)
      @show e¹ = norm(dir[coarse]["u"] - ue, ord)  # Δx
      @show e² = norm(interp(dir[mid]["x"], dir[mid]["u"], x) - ue, ord)  # Δx / 2
      @show ord
      @show p = log(e¹ / e²) / log(beta)
    end
  end

  # gui()
  return
end

if (@__FILE__) ∈ ("string", abspath(PROGRAM_FILE))
  main()
end

#####################
# OBSOLETE/COMMENTS #
#####################
# @show v¹ = [δx, 0]
# @show y₂(s, o[1], o[1] + δx, o[2])
# @show v² = (0, y₂(s, o[1], o[1] + δx, o[2]))