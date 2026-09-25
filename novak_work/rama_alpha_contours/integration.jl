# This file is included inside RamaAlphaContours. The arrays below represent
# cell averages; direct point evaluations have a separate diagnostic API.

"""
    periodic_density_grid(; grid_size=96, grid_origin=(-180.0, -180.0))

Construct square cells on one 360-degree period. `grid_origin` is the lower-left
cell corner, reduced modulo 360 to [-180,180). Edges and centers remain ordered
on that unwrapped interval; they are never individually wrapped and sorted.
Density matrices are indexed `[psi, phi]`.
"""
function periodic_density_grid(; grid_size::Integer=96,
        grid_origin=(-180.0, -180.0))
    grid_size > 0 || throw(ArgumentError("grid_size must be positive"))
    length(grid_origin) == 2 || throw(ArgumentError("grid_origin must contain phi and psi"))
    all(x -> x isa Real && isfinite(x), grid_origin) ||
        throw(ArgumentError("grid_origin coordinates must be finite real numbers"))
    n = Int(grid_size)
    origin = (_wrap_angle(grid_origin[1]), _wrap_angle(grid_origin[2]))
    step = PERIOD / n
    phi_edges = collect(range(origin[1], origin[1] + PERIOD; length=n + 1))
    psi_edges = collect(range(origin[2], origin[2] + PERIOD; length=n + 1))
    phi_grid = [origin[1] + (i - 0.5) * step for i in 1:n]
    psi_grid = [origin[2] + (j - 0.5) * step for j in 1:n]
    return (; phi_edges, psi_edges, phi_grid, psi_grid, cell_area=step^2,
        grid_origin=origin)
end

# Three components carry (phi, psi, density) through every clipping step.
# A triangle clipped by a rectangle has at most seven vertices. Twelve slots
# allow harmless coincident vertices on clipping lines without growing buffers.
const _DensityVertex = NTuple{3,Float64}

@inline function _density_input_triangle_area(a, b, c)
    bx, by = b[1] - a[1], b[2] - a[2]
    cx, cy = c[1] - a[1], c[2] - a[2]
    determinant = bx * cy - by * cx
    if abs(determinant) <= 32eps(Float64) * (abs(bx * cy) + abs(by * cx))
        # Geometry already uses higher precision for almost collinear input.
        # Match that contract when checking the stored area, without calling
        # another module's private circumcircle implementation.
        return setprecision(BigFloat, 256) do
            ax, ay = BigFloat(a[1]), BigFloat(a[2])
            bx, by = BigFloat(b[1]) - ax, BigFloat(b[2]) - ay
            cx, cy = BigFloat(c[1]) - ax, BigFloat(c[2]) - ay
            Float64(abs(bx * cy - by * cx) / 2)
        end
    end
    return abs(determinant) / 2
end

function _check_triangle_density(triangles, point_density)
    isempty(triangles) && throw(ArgumentError("density triangles must not be empty"))
    isempty(point_density) && throw(ArgumentError("point densities must not be empty"))
    all(x -> isfinite(x) && x >= 0, point_density) ||
        throw(ArgumentError("point densities must be finite and nonnegative"))
    npoints = length(point_density)
    for tri in triangles
        all(i -> i isa Integer && 1 <= i <= npoints, tri.base) ||
            throw(ArgumentError("triangle vertex indices must address point_density"))
        all(isfinite, (tri.a..., tri.b..., tri.c..., tri.area)) ||
            throw(ArgumentError("triangle coordinates and areas must be finite"))
        area = _density_input_triangle_area(tri.a, tri.b, tri.c)
        tri.area > 0 && area > 0 ||
            throw(ArgumentError("density triangles must have positive area"))
        isapprox(area, tri.area; rtol=1e-8, atol=0.0) ||
            throw(ArgumentError("stored triangle area disagrees with its coordinates"))
    end
    return nothing
end

@inline function _clip_density_halfplane!(dest::Vector{_DensityVertex},
        source::Vector{_DensityVertex}, n::Int, axis::Int, bound::Float64,
        keep_above::Bool)
    n == 0 && return 0
    nout = 0
    previous = source[n]
    previous_inside = keep_above ? previous[axis] >= bound : previous[axis] <= bound
    @inbounds for k in 1:n
        current = source[k]
        current_inside = keep_above ? current[axis] >= bound : current[axis] <= bound
        if current_inside != previous_inside
            t = clamp((bound - previous[axis]) / (current[axis] - previous[axis]), 0.0, 1.0)
            x = muladd(t, current[1] - previous[1], previous[1])
            y = muladd(t, current[2] - previous[2], previous[2])
            # Anchor at the endpoint with larger weight. This preserves both
            # endpoint values and constants without subtractive loss when
            # neighboring vertices have very different positive densities.
            rho = t <= 0.5 ? muladd(t, current[3] - previous[3], previous[3]) :
                muladd(1 - t, previous[3] - current[3], current[3])
            # Pin the clipped coordinate to its boundary. Otherwise a rounded
            # intersection can drift across the next cell's matching boundary.
            vertex = axis == 1 ? (bound, y, rho) : (x, bound, rho)
            nout += 1
            dest[nout] = vertex
        end
        if current_inside
            nout += 1
            dest[nout] = current
        end
        previous = current
        previous_inside = current_inside
    end
    return nout
end

@inline function _density_polygon_integrals(vertices::Vector{_DensityVertex}, n::Int)
    n < 3 && return (0.0, 0.0)
    origin = vertices[1]
    area = 0.0
    mass = 0.0
    @inbounds for k in 2:(n - 1)
        b, c = vertices[k], vertices[k + 1]
        # Translate to the fan vertex before taking determinants. Global-angle
        # shoelace sums would lose accuracy for the smallest observed triangles.
        piece_area = abs((b[1] - origin[1]) * (c[2] - origin[2]) -
            (b[2] - origin[2]) * (c[1] - origin[1])) / 2
        area += piece_area
        mass += piece_area * ((origin[3] + b[3] + c[3]) / 3)
    end
    return area, mass
end

@inline function _deposit_density_triangle!(cell_mass, covered_area, buf1, buf2,
        a, b, c, da, db, dc, triangle_area, grid)
    n = length(grid.phi_grid)
    xlo, xhi = first(grid.phi_edges), last(grid.phi_edges)
    ylo, yhi = first(grid.psi_edges), last(grid.psi_edges)
    minx, maxx = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
    miny, maxy = min(a[2], b[2], c[2]), max(a[2], b[2], c[2])
    (maxx <= xlo || minx >= xhi || maxy <= ylo || miny >= yhi) && return nothing
    invstep = n / PERIOD
    i0 = clamp(floor(Int, (max(minx, xlo) - xlo) * invstep) + 1, 1, n)
    i1 = clamp(floor(Int, (min(maxx, xhi) - xlo) * invstep) + 1, 1, n)
    j0 = clamp(floor(Int, (max(miny, ylo) - ylo) * invstep) + 1, 1, n)
    j1 = clamp(floor(Int, (min(maxy, yhi) - ylo) * invstep) + 1, 1, n)
    if i0 == i1 && j0 == j1 && minx >= xlo && maxx <= xhi && miny >= ylo && maxy <= yhi
        # Most triangles in a large residue dataset lie wholly inside one cell.
        # This route needs no clipping and retains their exact linear-field mass.
        @inbounds covered_area[j0, i0] += triangle_area
        @inbounds cell_mass[j0, i0] += triangle_area * ((da + db + dc) / 3)
        return nothing
    end
    @inbounds for j in j0:j1, i in i0:i1
        buf1[1] = (a[1], a[2], da)
        buf1[2] = (b[1], b[2], db)
        buf1[3] = (c[1], c[2], dc)
        nv = _clip_density_halfplane!(buf2, buf1, 3, 1, grid.phi_edges[i], true)
        nv = _clip_density_halfplane!(buf1, buf2, nv, 1, grid.phi_edges[i + 1], false)
        nv = _clip_density_halfplane!(buf2, buf1, nv, 2, grid.psi_edges[j], true)
        nv = _clip_density_halfplane!(buf1, buf2, nv, 2, grid.psi_edges[j + 1], false)
        area, mass = _density_polygon_integrals(buf1, nv)
        covered_area[j, i] += area
        cell_mass[j, i] += mass
    end
    return nothing
end

"""
    integrate_delaunay_density(triangles, point_density;
        grid_size=96, grid_origin=(-180.0,-180.0), progress=_quiet_progress)

Integrate the periodic, piecewise-linear field over every grid cell, without
normalizing its mass. Each triangle is clipped to the cells it intersects;
linear interpolation carries density to the new polygon vertices. Polygon mass
is the sum of fan-triangle areas times their mean vertex density. Consequently
even a triangle missing every cell center contributes its full mass.

`density` contains cell averages, `cell_mass` contains integrals, and
`covered_area` audits the geometric partition. Partial or overlapping triangle
collections are accepted for diagnostics, but `coverage_valid` is false unless
every cell is covered once within a relative tolerance of 1e-8. That numerical
area check supplements, and does not replace, the triangulation certificate.
"""
function integrate_delaunay_density(triangles::AbstractVector,
        point_density::AbstractVector{<:Real}; grid_size::Integer=96,
        grid_origin=(-180.0, -180.0), progress=_quiet_progress)
    _check_triangle_density(triangles, point_density)
    grid = periodic_density_grid(; grid_size, grid_origin)
    n = length(grid.phi_grid)
    cell_mass = zeros(Float64, n, n)
    covered_area = zeros(Float64, n, n)
    buf1 = Vector{_DensityVertex}(undef, 12)
    buf2 = similar(buf1)
    exact_mass = 0.0
    exact_mass_correction = 0.0
    xlo, xhi = first(grid.phi_edges), last(grid.phi_edges)
    ylo, yhi = first(grid.psi_edges), last(grid.psi_edges)
    report_every = _progress_every(length(triangles))
    for (k, tri) in enumerate(triangles)
        da = Float64(point_density[tri.base[1]])
        db = Float64(point_density[tri.base[2]])
        dc = Float64(point_density[tri.base[3]])
        contribution = tri.area * ((da + db + dc) / 3) - exact_mass_correction
        next_mass = exact_mass + contribution
        exact_mass_correction = (next_mass - exact_mass) - contribution
        exact_mass = next_mass
        foreach_periodic_triangle(tri; xlimits=(xlo, xhi), ylimits=(ylo, yhi)) do a, b, c
            _deposit_density_triangle!(cell_mass, covered_area, buf1, buf2,
                a, b, c, da, db, dc, tri.area, grid)
        end
        k % report_every == 0 && progress("conservative density integration: $(k)/$(length(triangles)) triangles")
    end
    total_mass = sum(cell_mass)
    relative_mass_error = exact_mass == 0 ? abs(total_mass) : abs(total_mass - exact_mass) / exact_mass
    max_relative_coverage_error = maximum(abs(area / grid.cell_area - 1) for area in covered_area)
    return merge(grid, (; density=cell_mass ./ grid.cell_area, cell_mass,
        covered_area, total_mass, exact_mass, relative_mass_error,
        max_relative_coverage_error, coverage_valid=max_relative_coverage_error <= 1e-8,
        integration_method=:triangle_cell_exact))
end

# For a linear field, a cut below the middle vertex density removes a similar
# triangle around the minimum; a cut above it leaves one around the maximum.
# These formulas equal polygon clipping, without allocating polygons in each
# threshold search iteration. Equal vertex densities are handled by the end
# cases before either denominator is used.
@inline function _triangle_superlevel_integrals(area::Real, da::Float64,
        db::Float64, dc::Float64, level::Float64)
    low = min(da, db, dc)
    high = max(da, db, dc)
    full_mass = area * ((da + db + dc) / 3)
    level <= low && return (Float64(area), full_mass)
    level >= high && return (0.0, 0.0)
    middle = max(min(da, db), min(max(da, db), dc))
    if level <= middle
        fraction = ((level - low) / (middle - low)) * ((level - low) / (high - low))
        below_area = area * fraction
        return (area - below_area, max(full_mass - below_area * ((low + 2level) / 3), 0.0))
    else
        fraction = ((high - level) / (high - low)) * ((high - level) / (high - middle))
        above_area = area * fraction
        return (above_area, above_area * ((high + 2level) / 3))
    end
end

"""
    triangle_superlevel_summary(triangles, point_density, level)

Exact area and density mass of `{rho >= level}` for the unsmoothed linear field,
using one representative per periodic triangle. This calculation does not use a
grid. Flat triangles at the threshold are included in full, so plateaus can
prevent a level set from containing exactly a prescribed mass fraction.
"""
function triangle_superlevel_summary(triangles::AbstractVector,
        point_density::AbstractVector{<:Real}, level::Real)
    _check_triangle_density(triangles, point_density)
    isfinite(level) || throw(ArgumentError("density level must be finite"))
    threshold = Float64(level)
    area, mass, total_mass = 0.0, 0.0, 0.0
    @inbounds for tri in triangles
        da, db, dc = (Float64(point_density[i]) for i in tri.base)
        piece_area, piece_mass = _triangle_superlevel_integrals(tri.area, da, db, dc, threshold)
        area += piece_area
        mass += piece_mass
        total_mass += tri.area * ((da + db + dc) / 3)
    end
    return (; area, mass, total_mass,
        mass_fraction=total_mass == 0 ? 0.0 : mass / total_mass,
        area_fraction=area / PERIOD^2)
end

"""
    density_level_for_triangle_mass(triangles, point_density;
        mass_fraction=0.98, rtol=1e-10, maxiters=128)

Find the largest density threshold whose exact unsmoothed superlevel mass is at
least the requested fraction. Bisection stops when the threshold bracket is
within `rtol` of its current density scale. No grid is used. On flat-density
plateaus the selected region may exceed the requested mass; inspect
`triangle_superlevel_summary` to report what it actually contains.
"""
function density_level_for_triangle_mass(triangles::AbstractVector,
        point_density::AbstractVector{<:Real}; mass_fraction::Real=0.98,
        rtol::Real=1e-10, maxiters::Integer=128)
    _check_triangle_density(triangles, point_density)
    isfinite(mass_fraction) && 0 < mass_fraction <= 1 ||
        throw(ArgumentError("mass_fraction must lie in (0, 1]"))
    isfinite(rtol) && 0 < rtol < 1 || throw(ArgumentError("rtol must lie in (0, 1)"))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive"))
    lower, upper, total_mass = Inf, 0.0, 0.0
    @inbounds for tri in triangles
        da, db, dc = (Float64(point_density[i]) for i in tri.base)
        lower = min(lower, da, db, dc)
        upper = max(upper, da, db, dc)
        total_mass += tri.area * ((da + db + dc) / 3)
    end
    total_mass > 0 || throw(ArgumentError("density field has zero mass"))
    (mass_fraction == 1 || lower == upper) && return lower
    desired_mass = Float64(mass_fraction) * total_mass
    # A positive-mass plateau at the maximum may already meet the target.
    maximum_level_mass = 0.0
    @inbounds for tri in triangles
        da, db, dc = (Float64(point_density[i]) for i in tri.base)
        da == db == dc == upper && (maximum_level_mass += tri.area * upper)
    end
    maximum_level_mass >= desired_mass && return upper
    for _ in 1:Int(maxiters)
        # A maximum density far above the desired threshold must not impose
        # an absolute tolerance so large that a lower-density cut is lost.
        upper - lower <= Float64(rtol) * max(abs(lower), abs(upper)) && break
        midpoint = lower + (upper - lower) / 2
        (midpoint == lower || midpoint == upper) && break
        mass = 0.0
        @inbounds for tri in triangles
            da, db, dc = (Float64(point_density[i]) for i in tri.base)
            _, piece_mass = _triangle_superlevel_integrals(tri.area, da, db, dc, midpoint)
            mass += piece_mass
        end
        if mass >= desired_mass
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return lower
end

@inline function _density_point_weights(px::Float64, py::Float64, a, b, c)
    # The tolerances are in barycentric coordinates. An absolute area cutoff
    # would discard the very small triangles whose mass deposition fixes.
    px == a[1] && py == a[2] && return (1.0, 0.0, 0.0)
    px == b[1] && py == b[2] && return (0.0, 1.0, 0.0)
    px == c[1] && py == c[2] && return (0.0, 0.0, 1.0)
    den = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
    den == 0 && return nothing
    wb = ((px - a[1]) * (c[2] - a[2]) - (py - a[2]) * (c[1] - a[1])) / den
    wc = ((b[1] - a[1]) * (py - a[2]) - (b[2] - a[2]) * (px - a[1])) / den
    wa = 1 - wb - wc
    min(wa, wb, wc) >= -1e-10 || return nothing
    wa, wb, wc = max(wa, 0.0), max(wb, 0.0), max(wc, 0.0)
    total = wa + wb + wc
    return (wa / total, wb / total, wc / total)
end

@inline function _density_point_value(weights, da, db, dc)
    # Preserve inclusive constant plateaus exactly. In the nonconstant case,
    # nonnegative weighted sums avoid cancellation at a low-density vertex
    # beside a much larger one; a fixed-corner difference formula would not.
    da == db == dc && return da
    wa, wb, wc = weights
    return wa * da + wb * db + wc * dc
end

"""
    sample_triangle_density(triangles, point_density, phi_grid, psi_grid)

Evaluate the original linear field at physical query locations on a rectangular
grid. Return `(density, covered)`. This is a point-evaluation oracle for plots
and comparisons; its samples must not replace conservative cell masses in the
production density grid. Uncovered samples are zero and marked false. A maximum
resolves overlaps deterministically for deliberately uncertified diagnostics.
"""
function sample_triangle_density(triangles::AbstractVector,
        point_density::AbstractVector{<:Real}, phi_grid::AbstractVector{<:Real},
        psi_grid::AbstractVector{<:Real}; progress=_quiet_progress)
    _check_triangle_density(triangles, point_density)
    for grid in (phi_grid, psi_grid)
        !isempty(grid) && all(isfinite, grid) && issorted(grid) &&
            all(>(0), diff(grid)) && last(grid) - first(grid) < PERIOD ||
            throw(ArgumentError("sample coordinates must be finite, strictly increasing, and span less than 360 degrees"))
    end
    density = zeros(Float64, length(psi_grid), length(phi_grid))
    covered = falses(size(density))
    xlimits, ylimits = (first(phi_grid), last(phi_grid)), (first(psi_grid), last(psi_grid))
    report_every = _progress_every(length(triangles))
    for (k, tri) in enumerate(triangles)
        da, db, dc = (Float64(point_density[i]) for i in tri.base)
        foreach_periodic_triangle(tri; xlimits, ylimits) do a, b, c
            ir = _grid_index_range(phi_grid, min(a[1], b[1], c[1]), max(a[1], b[1], c[1]))
            jr = _grid_index_range(psi_grid, min(a[2], b[2], c[2]), max(a[2], b[2], c[2]))
            @inbounds for j in jr, i in ir
                weights = _density_point_weights(Float64(phi_grid[i]), Float64(psi_grid[j]), a, b, c)
                weights === nothing && continue
                value = _density_point_value(weights, da, db, dc)
                density[j, i] = covered[j, i] ? max(density[j, i], value) : value
                covered[j, i] = true
            end
        end
        k % report_every == 0 && progress("direct density point evaluation: $(k)/$(length(triangles)) triangles")
    end
    return density, covered
end

"""
    empirical_superlevel_fraction(points, weights, triangles, point_density,
        level; bucket_grid=64)

Fraction of weighted observations in the unsmoothed region `{rho >= level}`.
Membership uses barycentric point evaluation in the original triangles, not a
raster mask or a list of selected vertices. Periodic point buckets avoid testing
every observation against every triangle. This is in-sample coverage if these
are the observations used to construct the density.
"""
function empirical_superlevel_fraction(points::AbstractMatrix{<:Real},
        weights::AbstractVector{<:Real}, triangles::AbstractVector,
        point_density::AbstractVector{<:Real}, level::Real; bucket_grid::Integer=64)
    _check_points(points)
    _check_triangle_density(triangles, point_density)
    length(weights) == size(points, 1) || throw(DimensionMismatch("one weight is required per observation"))
    all(w -> isfinite(w) && w >= 0, weights) && sum(weights) > 0 ||
        throw(ArgumentError("observation weights must be finite, nonnegative, and have positive sum"))
    isfinite(level) || throw(ArgumentError("density level must be finite"))
    bucket_grid > 0 || throw(ArgumentError("bucket_grid must be positive"))
    n = Int(bucket_grid)
    np = size(points, 1)
    heads = zeros(Int, n, n)
    next = zeros(Int, np)
    phi = Vector{Float64}(undef, np)
    psi = similar(phi)
    included = falses(np)
    @inbounds for k in 1:np
        phi[k], psi[k] = _wrap_angle(points[k, 1]), _wrap_angle(points[k, 2])
        i, j = _grid_bin_index(phi[k], n), _grid_bin_index(psi[k], n)
        next[k], heads[j, i] = heads[j, i], k
    end
    threshold = Float64(level)
    invstep = n / PERIOD
    for tri in triangles
        da, db, dc = (Float64(point_density[i]) for i in tri.base)
        max(da, db, dc) < threshold && continue
        foreach_periodic_triangle(tri) do a, b, c
            minx = max(min(a[1], b[1], c[1]), DOMAIN_MIN)
            maxx = min(max(a[1], b[1], c[1]), DOMAIN_MAX)
            miny = max(min(a[2], b[2], c[2]), DOMAIN_MIN)
            maxy = min(max(a[2], b[2], c[2]), DOMAIN_MAX)
            (minx > maxx || miny > maxy) && return nothing
            i0 = clamp(floor(Int, (minx - DOMAIN_MIN) * invstep) + 1, 1, n)
            i1 = clamp(floor(Int, (maxx - DOMAIN_MIN) * invstep) + 1, 1, n)
            j0 = clamp(floor(Int, (miny - DOMAIN_MIN) * invstep) + 1, 1, n)
            j1 = clamp(floor(Int, (maxy - DOMAIN_MIN) * invstep) + 1, 1, n)
            @inbounds for j in j0:j1, i in i0:i1
                k = heads[j, i]
                while k != 0
                    if !included[k]
                        barycentric = _density_point_weights(phi[k], psi[k], a, b, c)
                        if barycentric !== nothing
                            value = _density_point_value(barycentric, da, db, dc)
                            included[k] = value >= threshold
                        end
                    end
                    k = next[k]
                end
            end
        end
    end
    return sum((weights[k] for k in eachindex(weights) if included[k]); init=0.0) / sum(weights)
end
