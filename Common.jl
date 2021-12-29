module Common
  using Unroll
  using FFTW
  export compute_l2norm, compute_l2norm_bnds, compute_residual
  export tdms, tdma, wcL, wcR, crwcL, crwcR
  export wenoL, wenoR, wenoL_roe, wenoR_roe, fluxes_roe
  export restriction, prolongation, gauss_seidel_mg
  export fps, vm_rhs, wavespace, vm_ic

  #=
  # export all symbols from this module
  # ref: discourse.julialang.org/t/exportall/4970/16
  # ==> does not work !!
  for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
      @eval export $n
    end
  end
  =#

  restriction(nxf, nyf, nxc, nyc, r, ec) = begin
    @fastmath @simd for j ∈ 2:nyc for i ∈ 2:nxc
      # grid index for fine grid for the same coarse point
      center = 4r[2i-1, 2j-1]
      # E, W, N, S with respect to coarse grid point in fine grid
      grid = 2(r[2i-1, 2j-1+1] + r[2i-1, 2j-1-1] + r[2i-1+1, 2j-1] + r[2i-1-1, 2j-1])
      # NE, NW, SE, SW with respect to coarse grid point in fine grid
      corner = r[2i-1+1, 2j-1+1] + r[2i-1+1, 2j-1-1] + r[2i-1-1, 2j-1+1] + r[2i-1-1, 2j-1-1]
      # restriction using trapezoidal rule
      ec[i, j] = (center + grid + corner) / 16
    end end

    # restriction for boundary points bottom and top
    @fastmath @simd for j ∈ 1:nyc + 1
      # bottom boundary i = 1
      ec[1, j] = r[1, 2j-1]
      # top boundary i = ny_coarse+1
      ec[nxc+1, j] = r[nxf+1, 2j-1]
    end

    # restriction for boundary poinys left and right
    @fastmath @simd for i ∈ 1:nxc + 1
      # left boundary j = 1
      ec[i, 1] = r[2i-1, 1]
      # right boundary nx_coarse+1
      ec[i, nyc+1] = r[2i-1, nyf+1]
    end
  end

  prolongation(nxc, nyc, nxf, nyf, unc, ef) = begin
    @fastmath @simd for j ∈ 1:nyc for i ∈ 1:nxc
      # direct injection at center point
      ef[2i-1, 2j-1] = unc[i, j]
      # east neighnour on fine grid corresponding to coarse grid point
      ef[2i-1, 2j-1+1] = .5(unc[i, j] + unc[i, j+1])
      # north neighbout on fine grid corresponding to coarse grid point
      ef[2i-1+1, 2j-1] = .5(unc[i, j] + unc[i+1, j])
      # NE neighbour on fine grid corresponding to coarse grid point
      ef[2i-1+1, 2j-1+1] = .25(unc[i, j] + unc[i, j+1] + unc[i+1, j] + unc[i+1, j+1])
    end end

    # update boundary points
    @fastmath @simd for i ∈ 1:nxc + 1
      # left boundary j = 1
      ef[2i-1, 1] = unc[i, 1]
      # right boundary j = nx_fine + 1
      ef[2i-1, nyf+1] = unc[i, nyc+1]
    end

    @fastmath @simd for j ∈ 1:nyc + 1
      # bottom boundary i = 1
      ef[1, 2j-1] = unc[1, j]
      # top boundary i = ny_fine + 1
      ef[nxf+1, 2j-1] = unc[nxc+1, j]
    end
  end

  gauss_seidel_mg(nx, ny, Δx, Δy, f, un, V) = begin
    rt = zeros(Float64, nx + 1, ny + 1)
    # docs.julialang.org/en/v1/manual/performance-tips/#man-performance-annotations
    for it ∈ 1:V
      # compute solution at next time step ϕ^(k+1) = ϕ^k + ωr^(k+1)
      @fastmath @simd for j ∈ 2:ny for i ∈ 2:nx
        rt[i, j] = f[i, j] - (
          (un[i+1, j] - 2un[i, j] + un[i-1, j]) / Δx^2 +
          (un[i, j+1] - 2un[i, j] + un[i, j-1]) / Δy^2
        )
        un[i, j] += rt[i, j] / (-2 / Δx^2 - 2 / Δy^2)
      end end
    end
    # return  # not mandatory with end
  end

  # -----------------------------------------------------------------------------#
  # Fast poisson solver for periodic domain
  # -----------------------------------------------------------------------------#
  fps(nx, ny, Δx, Δy, u, e, data, data1, f, s, ε=1.e-6) = begin
    kx = Array{Float64}(undef, nx)
    ky = Array{Float64}(undef, ny)

    aa = -2 / Δx^2 - 2 / Δy^2
    bb = 2 / Δx^2
    cc = 2 / Δy^2

    # wave number indexing
    hx = 2π / nx

    @fastmath @simd for i ∈ 1:nx÷2
      kx[i] = hx * (i - 1)
      kx[i+nx÷2] = hx * (i - nx ÷ 2 - 1)
    end
    kx[1] = ε
    ky = kx

    @unroll data[1:nx, 1:ny] = complex(f[1:nx, 1:ny], 0.)

    e = fft(data)
    e[1, 1] = 0
    @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
      data1[i, j] = e[i, j] / (aa + bb * cos(kx[i]) + cc * cos(ky[j]))
    end end

    s[2:nx+1, 2:ny+1] = real(ifft(data1))
    return
  end


  # -----------------------------------------------------------------------------#
  # Calculate right hand term of the inviscid Burgers equation
  # r = -J(w,ψ) + ν ∇^2(w)
  # -----------------------------------------------------------------------------#
  vm_rhs(nx, ny, Δx, Δy, re, w, u, e, data, data1, r, s, f) = begin
    # compute streamfunction from vorticity
    @unroll f[1:nx, 1:ny] = -w[2:nx+1, 2:ny+1]

    fps(nx, ny, Δx, Δy, u, e, data, data1, f, s)

    @views begin
      # periodic BC
      s[nx+2, :] = s[2, :]
      s[:, ny+2] = s[:, 2]

      # ghost points
      s[1, :] = s[nx+1, :]
      s[:, 1] = s[:, ny+1]
    end

    # Arakawa numerical scheme for Jacobian
    aa = 1 / (re * Δx^2)
    bb = 1 / (re * Δy^2)
    gg = 1 / (4Δx * Δy)
    hh = 1 / 3

    @fastmath @simd for j ∈ 2:ny + 1 for i ∈ 2:nx + 1
      j1 = (
        (w[i+1, j] - w[i-1, j]) * (s[i, j+1] - s[i, j-1]) -
        (w[i, j+1] - w[i, j-1]) * (s[i+1, j] - s[i-1, j])
      )

      j2 = (
        w[i+1, j] * (s[i+1, j+1] - s[i+1, j-1]) -
        w[i-1, j] * (s[i-1, j+1] - s[i-1, j-1]) -
        w[i, j+1] * (s[i+1, j+1] - s[i-1, j+1]) +
        w[i, j-1] * (s[i+1, j-1] - s[i-1, j-1])
      )

      j3 = (
        w[i+1, j+1] * (s[i, j+1] - s[i+1, j]) -
        w[i-1, j-1] * (s[i-1, j] - s[i, j-1]) -
        w[i-1, j+1] * (s[i, j+1] - s[i-1, j]) +
        w[i+1, j-1] * (s[i+1, j] - s[i, j-1])
      )

      jac = gg * (j1 + j2 + j3) * hh

      # Central difference for Laplacian
      r[i, j] = -jac + (
        aa * (w[i+1, j] - 2w[i, j] + w[i-1, j]) +
        bb * (w[i, j+1] - 2w[i, j] + w[i, j-1])
      )
    end end
  end

  wavespace(nx, ny, Δx, Δy, ε=1e-6) = begin
    kx = Array{Float64}(undef, nx)
    ky = Array{Float64}(undef, ny)
    k2 = Array{Float64}(undef, nx, ny)

    # wave number indexing
    hx = 2π / (nx * Δx)

    @fastmath @simd for i ∈ 1:nx÷2
      kx[i] = hx * (i - 1.)
      kx[i+nx÷2] = hx * (i - nx ÷ 2 - 1)
    end
    kx[1] = ε
    ky = kx

    @fastmath @simd for j ∈ 1:ny for i ∈ 1:nx
      k2[i, j] = kx[i]^2 + ky[j]^2
    end end

    return k2
  end


  # initial condition for vortex merger problem
  vm_ic(nx, ny, x, y, w) = begin
    σ = π
    xc1, yc1 = π - π / 4., π
    xc2, yc2 = π + π / 4., π

    @fastmath @simd for j ∈ 2:ny + 2 for i ∈ 2:nx + 2
      w[i, j] = (
        exp(-σ * ((x[i-1] - xc1)^2 + (y[j-1] - yc1)^2)) +
        exp(-σ * ((x[i-1] - xc2)^2 + (y[j-1] - yc2)^2))
      )
    end end
  end

  # -----------------------------------------------------------------------------#
  # Compute L-2 norm for a vector
  # -----------------------------------------------------------------------------#
  compute_l2norm(nx, r) = begin
    rms = eltype(r)(0.); @unroll rms += r[2:nx]^2
    return √(rms / (nx - 1))
  end

  compute_l2norm(nx, ny, r) = begin
    rms = eltype(r)(0.); @unroll rms += r[2:nx, 2:ny]^2
    return √(rms / ((nx - 1) * (ny - 1)))
  end

  compute_l2norm_bnds(nx, ny, r) = begin
    rms = eltype(r)(0.); @unroll rms += r[1:nx+1, 1:ny+1]^2
    return √(rms / ((nx + 1) * (ny + 1)))
  end

  compute_residual(nx, ny, Δx, Δy, f, u_n, r) = begin
    @fastmath @simd for j ∈ 2:ny for i ∈ 2:nx
      r[i, j] = f[i, j] - (
        (u_n[i+1, j] - 2u_n[i, j] + u_n[i-1, j]) / Δx^2 +
        (u_n[i, j+1] - 2u_n[i, j] + u_n[i, j-1]) / Δy^2
      )
    end end
  end

"""
Solution to tridigonal system using Thomas algorithm (part of ctdms)
a: sub-diagonal
b: diagonal
r: rhs
x: solution
z: scratch array
s, e: start, end (solved points)
"""
  tdms(a, b, c, r, x, z, s, e) = begin  # TriDiagonal Matrix Solve ?
    # γ = Array{Float64}(undef, e)  # scratch array
    β = b[s]
    x[s] = r[s] / β

    @fastmath for i ∈ s + 1:e  # forward elimination, non-simd !
      z[i] = c[i-1] / β
      β = b[i] - a[i] * z[i]
      x[i] = (r[i] - a[i] * x[i-1]) / β
    end

    @fastmath for i ∈ e - 1:-1:s  # back-substitution, non-simd !
      x[i] -= z[i+1] * x[i+1]
    end
  end

  # -----------------------------------------------------------------------------#
  # Solution to tridigonal system using Thomas algorithm
  # -----------------------------------------------------------------------------#
  tdma(a, b, c, r, x, s, e) = begin  # TriDiagonal Matrix Algorithm
    @fastmath for i ∈ s + 1:e + 1  # non-simd !
      b[i] -= a[i] * (c[i-1] / b[i-1])
      r[i] -= a[i] * (r[i-1] / b[i-1])
    end

    x[e+1] = r[e+1] / b[e+1]

    @fastmath for i ∈ e:-1:s  # non-simd !
      x[i] = (r[i] - c[i] * x[i+1]) / b[i]
    end
  end

  # ---------------------------------------------------------------------------#
  # nonlinear weights for upwind direction
  # ---------------------------------------------------------------------------#
  @fastmath @inline wcL(v1, v2, v3, v4, v5, ε) = begin
    # smoothness indicators
    s1 = (13 / 12) * (v1 - 2v2 + v3)^2 + .25(v1 - 4v2 + 3v3)^2
    s2 = (13 / 12) * (v2 - 2v3 + v4)^2 + .25(v2 - v4)^2
    s3 = (13 / 12) * (v3 - 2v4 + v5)^2 + .25(3v3 - 4v4 + v5)^2

    # computing nonlinear weights w1,w2,w3
    c1 = .1 / (ε + s1)^2
    c2 = .6 / (ε + s2)^2
    c3 = .3 / (ε + s3)^2

    w1 = c1 / (c1 + c2 + c3)
    w2 = c2 / (c1 + c2 + c3)
    w3 = c3 / (c1 + c2 + c3)

    # candidate stencils
    q1 = +v1 / 3 - (7 / 6) * v2 + (11 / 6) * v3
    q2 = -v2 / 6 + (5 / 6) * v3 + v4 / 3
    q3 = +v3 / 3 + (5 / 6) * v4 - v5 / 6

    # reconstructed value at interface
    return w1 * q1 + w2 * q2 + w3 * q3
  end

  # ---------------------------------------------------------------------------#
  # nonlinear weights for downwind direction
  # ---------------------------------------------------------------------------#
  @fastmath @inline wcR(v1, v2, v3, v4, v5, ε) = begin
    s1 = (13 / 12) * (v1 - 2v2 + v3)^2 + .25(v1 - 4v2 + 3v3)^2
    s2 = (13 / 12) * (v2 - 2v3 + v4)^2 + .25(v2 - v4)^2
    s3 = (13 / 12) * (v3 - 2v4 + v5)^2 + .25(3v3 - 4v4 + v5)^2

    c1 = .3 / (ε + s1)^2
    c2 = .6 / (ε + s2)^2
    c3 = .1 / (ε + s3)^2

    w1 = c1 / (c1 + c2 + c3)
    w2 = c2 / (c1 + c2 + c3)
    w3 = c3 / (c1 + c2 + c3)

    # candidate stencils
    q1 = -v1 / 6 + (5 / 6) * v2 + v3 / 3
    q2 = +v2 / 3 + (5 / 6) * v3 - v4 / 6
    q3 = (11 / 6) * v3 - (7 / 6) * v4 + v5 / 3

    # reconstructed value at interface
    return w1 * q1 + w2 * q2 + w3 * q3
  end

  # ---------------------------------------------------------------------------#
  # nonlinear weights for upwind direction
  # ---------------------------------------------------------------------------#
  @fastmath @inline crwcL(v1, v2, v3, v4, v5, ε) = begin
    s1 = (13 / 12) * (v1 - 2v2 + v3)^2 + .25(v1 - 4v2 + 3v3)^2
    s2 = (13 / 12) * (v2 - 2v3 + v4)^2 + .25(v2 - v4)^2
    s3 = (13 / 12) * (v3 - 2v4 + v5)^2 + .25(3v3 - 4v4 + v5)^2

    c1 = .2 / (ε + s1)^2
    c2 = .5 / (ε + s2)^2
    c3 = .3 / (ε + s3)^2

    w1 = c1 / (c1 + c2 + c3)
    w2 = c2 / (c1 + c2 + c3)
    w3 = c3 / (c1 + c2 + c3)

    a1 = (2w1 + w2) / 3
    a2 = (w1 + 2w2 + 2w3) / 3
    a3 = w3 / 3.

    b1 = w1 / 6.
    b2 = (5w1 + 5w2 + w3) / 6
    b3 = (w2 + 5w3) / 6

    return a1, a2, a3, b1, b2, b3
  end

  # ---------------------------------------------------------------------------#
  # nonlinear weights for downwind direction
  # ---------------------------------------------------------------------------#
  @fastmath @inline crwcR(v1, v2, v3, v4, v5, ε) = begin
    s1 = (13 / 12) * (v1 - 2v2 + v3)^2 + .25(v1 - 4v2 + 3v3)^2
    s2 = (13 / 12) * (v2 - 2v3 + v4)^2 + .25(v2 - v4)^2
    s3 = (13 / 12) * (v3 - 2v4 + v5)^2 + .25(3v3 - 4v4 + v5)^2

    c1 = .3 / (ε + s1)^2
    c2 = .5 / (ε + s2)^2
    c3 = .2 / (ε + s3)^2

    w1 = c1 / (c1 + c2 + c3)
    w2 = c2 / (c1 + c2 + c3)
    w3 = c3 / (c1 + c2 + c3)

    a1 = w1 / 3
    a2 = (w3 + 2w2 + 2w1) / 3
    a3 = (2w3 + w2) / 3

    b1 = (w2 + 5w1) / 6
    b2 = (5w3 + 5w2 + w1) / 6
    b3 = w3 / 6

    return a1, a2, a3, b1, b2, b3
  end

  # -----------------------------------------------------------------------------#
  # WENO reconstruction for upwind direction (positive; left to right)
  # u(i): solution values at finite difference grid nodes i = 1,...,N
  # f(j): reconstructed values at nodes j = i-1/2; j = 1,...,N+1
  # -----------------------------------------------------------------------------#
  wenoL(n, u, f, ε=1e-6) = begin
    i = 0; f[i+1] = wcL(
      u[n-2],
      u[n-1],
      u[n],
      u[i+1],
      u[i+2], ε
    )

    i = 1; f[i+1] = wcL(
      u[n-1],
      u[n],
      u[i],
      u[i+1],
      u[i+2], ε
    )

    i = 2; f[i+1] = wcL(
      u[n],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )

    @simd for i ∈ 3:n - 2
      f[i+1] = wcL(
        u[i-2],
        u[i-1],
        u[i],
        u[i+1],
        u[i+2], ε
      )
    end

    i = n - 1; f[i+1] = wcL(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[1], ε
    )

    i = n; f[i+1] = wcL(
      u[i-2],
      u[i-1],
      u[i],
      u[1],
      u[2], ε
    )
    return
  end

  # -----------------------------------------------------------------------------#
  # WENO reconstruction for downwind direction (negative; right to left)
  # u(i): solution values at finite difference grid nodes i = 1,...,N+1
  # f(j): reconstructed values at nodes j = i-1/2; j = 2,...,N+1
  # -----------------------------------------------------------------------------#
  wenoR(n, u, f, ε=1e-6) = begin
    i = 1; f[i] = wcR(
      u[n-1],
      u[n],
      u[i],
      u[i+1],
      u[i+2], ε
    )

    i = 2; f[i] = wcR(
      u[n],
      u[i-1],
      u[i],
      u[i+1],
      u[i+2], ε
    )

    @simd for i ∈ 3:n - 2
      f[i] = wcR(
        u[i-2],
        u[i-1],
        u[i],
        u[i+1],
        u[i+2], ε
      )
    end

    i = n - 1; f[i] = wcR(
      u[i-2],
      u[i-1],
      u[i],
      u[i+1],
      u[1], ε
    )

    i = n; f[i] = wcR(
      u[i-2],
      u[i-1],
      u[i],
      u[1],
      u[2], ε
    )

    i = n + 1; f[i] = wcR(
      u[i-2],
      u[i-1],
      u[1],
      u[2],
      u[3], ε
    )
    return
  end

  # -----------------------------------------------------------------------------#
  # WENO reconstruction for upwind direction (positive; left to right)
  # u(i): solution values at finite difference grid nodes i = 1,...,N
  # f(j): reconstructed values at nodes j = i-1/2; j = 1,...,N+1
  # -----------------------------------------------------------------------------#
  wenoL_roe(n, u, f, ε=1e-6) = begin
    for m ∈ 1:3
      i = 0; f[i+1, m] = wcL(
        u[i+3, m],
        u[i+2, m],
        u[i+1, m],
        u[i+1, m],
        u[i+2, m], ε
      )

      i = 1; f[i+1, m] = wcL(
        u[i+1, m],
        u[i, m],
        u[i, m],
        u[i+1, m],
        u[i+2, m], ε
      )

      i = 2; f[i+1, m] = wcL(
        u[i-1, m],
        u[i-1, m],
        u[i, m],
        u[i+1, m],
        u[i+2, m], ε
      )

      @simd for i ∈ 3:n - 2
        f[i+1, m] = wcL(
          u[i-2, m],
          u[i-1, m],
          u[i, m],
          u[i+1, m],
          u[i+2, m], ε
        )
      end

      i = n - 1; f[i+1, m] = wcL(
        u[i-2, m],
        u[i-1, m],
        u[i, m],
        u[i+1, m],
        u[i+1, m], ε
      )

      i = n; f[i+1, m] = wcL(
        u[i-2, m],
        u[i-1, m],
        u[i, m],
        u[i, m],
        u[i-1, m], ε
      )
    end
    return
  end

  # -----------------------------------------------------------------------------#
  # WENO reconstruction for downwind direction (negative; right to left)
  # u(i): solution values at finite difference grid nodes i = 1,...,N+1
  # f(j): reconstructed values at nodes j = i-1/2; j = 2,...,N+1
  # -----------------------------------------------------------------------------#
  wenoR_roe(n, u, f, ε=1e-6) = begin
    for m ∈ 1:3
      i = 1; f[i, m] = wcR(
        u[i+1, m],
        u[i, m],
        u[i, m],
        u[i+1, m],
        u[i+2, m], ε
      )

      i = 2; f[i, m] = wcR(
        u[i-1, m],
        u[i-1, m],
        u[i, m],
        u[i+1, m],
        u[i+2, m], ε
      )

      @simd for i ∈ 3:n - 2
        f[i, m] = wcR(
          u[i-2, m],
          u[i-1, m],
          u[i, m],
          u[i+1, m],
          u[i+2, m], ε
        )
      end

      i = n - 1; f[i, m] = wcR(
        u[i-2, m],
        u[i-1, m],
        u[i, m],
        u[i+1, m],
        u[i+1, m], ε
      )

      i = n; f[i, m] = wcR(
        u[i-2, m],
        u[i-1, m],
        u[i, m],
        u[i, m],
        u[i-1, m], ε
      )

      i = n + 1; f[i, m] = wcR(
        u[i-2, m],
        u[i-1, m],
        u[i-1, m],
        u[i-2, m],
        u[i-3, m], ε
      )
    end
    return
  end

  # -----------------------------------------------------------------------------#
  # Calculate fluxes
  # -----------------------------------------------------------------------------#
  fluxes_roe(nx, γ, q, f) = begin
    @fastmath @simd for i ∈ 1:nx + 1
      p = (γ - 1) * (q[i, 3] - .5q[i, 2]^2 / q[i, 1])
      f[i, 1] = q[i, 2]
      f[i, 2] = q[i, 2] * q[i, 2] / q[i, 1] + p
      f[i, 3] = q[i, 2] * q[i, 3] / q[i, 1] + p * q[i, 2] / q[i, 1]
    end
  end
end
