"""
Numerical work shared by the density-adaptive alpha notebook, its tests, and its
reference comparisons. Angles are in degrees; density arrays use [psi, phi].
The notebook explains the protocol; this module keeps its numerical work testable.
"""
module RamaAlphaContours

using Statistics
using Serialization
using SHA
using Random
using ..RamaAlphaGeometry: DensityTriangle, triangle_circumradius,
    foreach_periodic_triangle, periodic_delaunay_density_triangles
import ..RamaAlphaGeometry

_quiet_progress(message) = nothing

function _check_points(points)
    size(points, 2) == 2 || throw(ArgumentError("points must have two columns: phi and psi"))
    size(points, 1) > 0 || throw(ArgumentError("point set must not be empty"))
    all(isfinite, points) || throw(ArgumentError("point coordinates must be finite"))
end

@inline _wrap_angle(x::Real) = mod(Float64(x) + 180.0, 360.0) - 180.0

const DOMAIN_MIN = -180.0
const DOMAIN_MAX = 180.0
const PERIOD = DOMAIN_MAX - DOMAIN_MIN

function _progress_every(total::Integer; steps::Integer=10)
    total_i = max(Int(total), 1)
    steps_i = max(Int(steps), 1)
    return max(1, cld(total_i, steps_i))
end

@inline function _grid_bin_index(angle::Real, n::Int)
    x = mod(Float64(angle) - DOMAIN_MIN, PERIOD)
    idx = Int(floor(x / PERIOD * n)) + 1
    return idx > n ? n : idx
end

function unique_points_with_counts(points::AbstractMatrix{<:Real})
    _check_points(points)
    index = Dict{Tuple{Float64,Float64},Int}()
    unique_pts = NTuple{2,Float64}[]
    counts = Int[]
    @inbounds for r in axes(points, 1)
        key = (_wrap_angle(points[r, 1]), _wrap_angle(points[r, 2]))
        k = get(index, key, 0)
        if k == 0
            push!(unique_pts, key)
            push!(counts, 1)
            index[key] = length(unique_pts)
        else
            counts[k] += 1
        end
    end
    mat = Matrix{Float64}(undef, length(unique_pts), 2)
    @inbounds for i in eachindex(unique_pts)
        mat[i, 1] = unique_pts[i][1]
        mat[i, 2] = unique_pts[i][2]
    end
    return mat, counts
end

function grid_representatives_with_counts(points::AbstractMatrix{<:Real}; grid_size::Integer=256)
    _check_points(points)
    n = Int(grid_size)
    n > 0 || throw(ArgumentError("grid_size must be positive."))
    counts = Dict{Tuple{Int,Int},Int}()
    @inbounds for r in axes(points, 1)
        i = _grid_bin_index(points[r, 1], n)
        j = _grid_bin_index(points[r, 2], n)
        key = (i, j)
        counts[key] = get(counts, key, 0) + 1
    end
    keys_sorted = sort!(collect(keys(counts)))
    reps = Matrix{Float64}(undef, length(keys_sorted), 2)
    multiplicities = Vector{Int}(undef, length(keys_sorted))
    step = PERIOD / n
    @inbounds for (k, (i, j)) in enumerate(keys_sorted)
        reps[k, 1] = DOMAIN_MIN + (i - 0.5) * step
        reps[k, 2] = DOMAIN_MIN + (j - 0.5) * step
        multiplicities[k] = counts[(i, j)]
    end
    return reps, multiplicities
end

function delaunay_input_points_with_counts(points::AbstractMatrix{<:Real};
        use_grid_representatives::Bool=false, representative_grid::Int=256,
        progress=_quiet_progress)
    n_raw = size(points, 1)
    if use_grid_representatives
        progress("compressing $(n_raw) raw points into weighted $(representative_grid)x$(representative_grid) grid representatives")
        reps, multiplicities = grid_representatives_with_counts(points; grid_size=representative_grid)
        ratio = n_raw / max(size(reps, 1), 1)
        summary = (;
            mode = :grid_representatives,
            representative_grid = representative_grid,
            n_raw_points = n_raw,
            n_representative_points = size(reps, 1),
            compression_ratio = ratio,
        )
        progress("Delaunay input ready: $(size(reps, 1)) weighted representatives; compression ratio=$(round(ratio; digits=2))")
        return reps, multiplicities, summary
    else
        progress("collapsing exact duplicate coordinates among $(n_raw) raw points")
        reps, multiplicities = unique_points_with_counts(points)
        ratio = n_raw / max(size(reps, 1), 1)
        summary = (;
            mode = :exact_unique_points,
            representative_grid = nothing,
            n_raw_points = n_raw,
            n_representative_points = size(reps, 1),
            compression_ratio = ratio,
        )
        progress("Delaunay input ready: $(size(reps, 1)) exact unique points; duplicate-collapse ratio=$(round(ratio; digits=2))")
        return reps, multiplicities, summary
    end
end

@inline function _euclidean_distance(a::NTuple{2,Float64}, b::NTuple{2,Float64})
    return hypot(a[1] - b[1], a[2] - b[2])
end

@inline function _signed_area2(px::Float64, py::Float64,
                              ax::Float64, ay::Float64,
                              bx::Float64, by::Float64)
    return (px - bx) * (ay - by) - (ax - bx) * (py - by)
end

@inline function _point_in_triangle(px::Float64, py::Float64,
                                   a::NTuple{2,Float64},
                                   b::NTuple{2,Float64},
                                   c::NTuple{2,Float64};
                                   tol::Float64=1e-10)
    d1 = _signed_area2(px, py, a[1], a[2], b[1], b[2])
    d2 = _signed_area2(px, py, b[1], b[2], c[1], c[2])
    d3 = _signed_area2(px, py, c[1], c[2], a[1], a[2])
    has_neg = (d1 < -tol) || (d2 < -tol) || (d3 < -tol)
    has_pos = (d1 > tol) || (d2 > tol) || (d3 > tol)
    return !(has_neg && has_pos)
end

function _grid_index_range(grid::AbstractVector{<:Real}, lo::Real, hi::Real)
    lo_eff = max(Float64(lo), Float64(first(grid)))
    hi_eff = min(Float64(hi), Float64(last(grid)))
    lo_eff > hi_eff && return 1:0
    i0 = searchsortedfirst(grid, lo_eff)
    i1 = searchsortedlast(grid, hi_eff)
    return i0:i1
end

function _grid_step(grid::AbstractVector{<:Real})
    return length(grid) > 1 ? abs(Float64(grid[2]) - Float64(grid[1])) : PERIOD
end

function _periodic_gaussian_kernel(step::Real, sigma::Real; cutoff_sigma::Real=4.0)
    sigma_f = Float64(sigma)
    sigma_f >= 0 || throw(ArgumentError("sigma must be nonnegative."))
    sigma_f == 0.0 && return (; offsets=[0], weights=[1.0])
    radius = max(1, ceil(Int, Float64(cutoff_sigma) * sigma_f / Float64(step)))
    offsets = collect(-radius:radius)
    weights = exp.(-((Float64.(offsets) .* Float64(step)).^2) ./ (2.0 * sigma_f^2))
    weights ./= sum(weights)
    return (; offsets, weights)
end

function _periodic_separable_gaussian_smooth(values::AbstractMatrix{<:Real},
                                            phi_grid::AbstractVector{<:Real},
                                            psi_grid::AbstractVector{<:Real};
                                            sigma_deg::Real)
    sigma_f = Float64(sigma_deg)
    isfinite(sigma_f) && sigma_f >= 0 || throw(ArgumentError("sigma_deg must be finite and nonnegative."))
    sigma_f == 0.0 && return Float64.(values)

    nrows, ncols = size(values)
    kx = _periodic_gaussian_kernel(_grid_step(phi_grid), sigma_f)
    ky = _periodic_gaussian_kernel(_grid_step(psi_grid), sigma_f)
    tmp = zeros(Float64, nrows, ncols)
    out = zeros(Float64, nrows, ncols)

    @inbounds for j in 1:nrows, i in 1:ncols
        acc = 0.0
        for q in eachindex(kx.offsets)
            acc += Float64(values[j, mod1(i - kx.offsets[q], ncols)]) * kx.weights[q]
        end
        tmp[j, i] = acc
    end

    @inbounds for j in 1:nrows, i in 1:ncols
        acc = 0.0
        for q in eachindex(ky.offsets)
            acc += tmp[mod1(j - ky.offsets[q], nrows), i] * ky.weights[q]
        end
        out[j, i] = acc
    end

    return out
end

"""
Smooth conservative cell averages with a periodic, unit-sum Gaussian stencil.
`density` remains a cell-average representation. `point_values` records the
pointwise representation used for level masks: the original linear field at
sigma=0, and the explicitly approximate bilinear reconstruction otherwise.
"""
function smooth_delaunay_density_estimate(est; sigma_deg::Real=8.0)
    smoothed = _periodic_separable_gaussian_smooth(est.density, est.phi_grid, est.psi_grid;
        sigma_deg=sigma_deg)
    exact_raw = sigma_deg == 0 && hasproperty(est, :triangles) && hasproperty(est, :point_density)
    point_values = if exact_raw
        values, covered = sample_triangle_density(est.triangles, est.point_density, est.phi_grid, est.psi_grid)
        all(covered) || error("piecewise-linear density has uncovered evaluation points")
        values
    else
        smoothed
    end
    before_mass = grid_density_mass(est)
    after_mass = sum(smoothed) * grid_cell_area(est)
    kx = _periodic_gaussian_kernel(_grid_step(est.phi_grid), sigma_deg)
    ky = _periodic_gaussian_kernel(_grid_step(est.psi_grid), sigma_deg)
    variance_x = sum(kx.weights .* (_grid_step(est.phi_grid) .* kx.offsets).^2)
    variance_y = sum(ky.weights .* (_grid_step(est.psi_grid) .* ky.offsets).^2)
    return merge(est, (;
        density=smoothed, cell_mass=smoothed .* grid_cell_area(est),
        point_values, raw_density=est.density,
        smoothing_sigma_deg=Float64(sigma_deg),
        reconstruction=exact_raw ? :piecewise_linear_delaunay : :periodic_bilinear_cell_averages,
        smoothing_summary=(sigma_deg=Float64(sigma_deg),
            kernel_variance_deg2=(variance_x, variance_y),
            mass_before=before_mass, mass_after=after_mass,
            relative_mass_error=before_mass == 0 ? abs(after_mass) : abs(after_mass-before_mass)/abs(before_mass)),
        method=sigma_deg == 0 ? :integrated_periodic_delaunay_area : :smoothed_integrated_periodic_delaunay_area,
    ))
end

@inline function _periodic_grid_position(x::Real, grid::AbstractVector{<:Real})
    n = length(grid)
    step = _grid_step(grid)
    s = mod(Float64(x) - Float64(first(grid)), PERIOD) / step
    flo = floor(Int, s)
    i0 = mod1(flo + 1, n)
    i1 = mod1(i0 + 1, n)
    return i0, i1, s - flo
end

function _periodic_bilinear_value(est, x::Real, y::Real; values=est.density)
    i0, i1, tx = _periodic_grid_position(x, est.phi_grid)
    j0, j1, ty = _periodic_grid_position(y, est.psi_grid)
    v00 = Float64(values[j0, i0])
    v10 = Float64(values[j0, i1])
    v01 = Float64(values[j1, i0])
    v11 = Float64(values[j1, i1])
    # Preserve a constant plateau at its inclusive density threshold exactly.
    v00 == v10 == v01 == v11 && return v00
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
end



function delaunay_point_density(area_acc::AbstractVector{<:Real},
                                multiplicities::AbstractVector{<:Integer};
                                area_floor_quantile::Real=0.01,
                                progress=_quiet_progress)
    length(area_acc) == length(multiplicities) || throw(DimensionMismatch("areas and multiplicities must have equal lengths"))
    all(x -> isfinite(x) && x >= 0, area_acc) || throw(ArgumentError("areas must be finite and nonnegative"))
    all(>(0), multiplicities) || throw(ArgumentError("multiplicities must be positive"))
    isfinite(area_floor_quantile) && 0 <= area_floor_quantile <= 1 || throw(ArgumentError("area_floor_quantile must lie in [0, 1]"))
    progress("computing Delaunay-area point densities for $(length(area_acc)) input points")
    positive = Float64[x for x in area_acc if x > 0]
    isempty(positive) && error("periodic Delaunay produced no positive point areas")
    length(positive) == length(area_acc) || error("periodic Delaunay produced zero point areas")
    area_floor = area_floor_quantile == 0 ? 0.0 : quantile(positive, Float64(area_floor_quantile))
    density = Vector{Float64}(undef, length(area_acc))
    @inbounds for i in eachindex(area_acc)
        density[i] = Float64(multiplicities[i]) / max(Float64(area_acc[i]), area_floor)
    end
    progress("point-density done: positive areas=$(length(positive)); density range=$(extrema(density))")
    observation_mass = Float64(sum(multiplicities))
    exact_mass = sum(Float64(area_acc[i]) * density[i] for i in eachindex(density))
    affected = count(<(area_floor), area_acc)
    affected_weight = sum(multiplicities[i] for i in eachindex(area_acc) if area_acc[i] < area_floor; init=0)
    return density, (; area_floor, area_floor_quantile=Float64(area_floor_quantile),
        area_min=minimum(positive), area_median=median(positive), area_max=maximum(positive),
        affected_vertices=affected, affected_vertex_fraction=affected / length(area_acc),
        affected_observation_fraction=affected_weight / observation_mass,
        observation_mass, exact_mass, mass_removed=observation_mass-exact_mass,
        removed_mass_fraction=(observation_mass-exact_mass)/observation_mass)
end

function grid_cell_area(est)
    dx = length(est.phi_grid) > 1 ? abs(est.phi_grid[2] - est.phi_grid[1]) : PERIOD
    dy = length(est.psi_grid) > 1 ? abs(est.psi_grid[2] - est.psi_grid[1]) : PERIOD
    return dx * dy
end

function grid_density_mass(est, mask::AbstractMatrix{Bool}=trues(size(est.density)))
    return sum(est.density[mask]) * grid_cell_area(est)
end

function density_level_for_grid_mass(est; mass_fraction::Real=0.98)
    f = Float64(mass_fraction)
    0 < f <= 1 || throw(ArgumentError("mass_fraction must lie in (0, 1]."))
    vals = collect(vec(est.density))
    sort!(vals; rev=true)
    cell_area = grid_cell_area(est)
    total = sum(vals) * cell_area
    total > 0 || error("density field has zero mass")
    acc = 0.0
    for v in vals
        acc += v * cell_area
        if acc / total >= f
            return v
        end
    end
    return vals[end]
end

"""Choose a mass-fraction level from the explicitly represented density field."""
function density_level_for_mass(est; mass_fraction::Real=0.98)
    if hasproperty(est, :reconstruction) && est.reconstruction == :piecewise_linear_delaunay
        return density_level_for_triangle_mass(est.triangles, est.point_density; mass_fraction=mass_fraction)
    end
    return density_level_for_grid_mass(est; mass_fraction=mass_fraction)
end

function alpha_triangle_grade(a::NTuple{2,Float64},
                              b::NTuple{2,Float64},
                              c::NTuple{2,Float64})
    eab = 0.5 * _euclidean_distance(a, b)
    eac = 0.5 * _euclidean_distance(a, c)
    ebc = 0.5 * _euclidean_distance(b, c)
    rad = triangle_circumradius(a, b, c)
    return max(eab, eac, ebc, rad)
end

const AlphaTriangle = NamedTuple{(:a, :b, :c, :grade, :gate_density),
    Tuple{NTuple{2,Float64}, NTuple{2,Float64}, NTuple{2,Float64}, Float64, Float64}}

function alpha_triangles_from_delaunay_density(triangles::AbstractVector{DensityTriangle},
                                               point_density::AbstractVector{<:Real};
                                               gate_estimate=nothing)
    rows = AlphaTriangle[]
    sizehint!(rows, length(triangles))
    @inbounds for tri in triangles
        ba, bb, bc = tri.base
        grade = alpha_triangle_grade(tri.a, tri.b, tri.c)
        isfinite(grade) || error("alpha geometry contains a triangle with nonfinite circumradius")
        if gate_estimate === nothing
            gate_density = min(Float64(point_density[ba]), Float64(point_density[bb]), Float64(point_density[bc]))
        else
            da = _periodic_bilinear_value(gate_estimate, tri.a[1], tri.a[2])
            db = _periodic_bilinear_value(gate_estimate, tri.b[1], tri.b[2])
            dc = _periodic_bilinear_value(gate_estimate, tri.c[1], tri.c[2])
            gate_density = min(da, db, dc)
        end
        push!(rows, (a=tri.a, b=tri.b, c=tri.c, grade=grade, gate_density=gate_density))
    end
    return rows
end

function mask_component_summary_torus(mask::AbstractMatrix{Bool})
    nrows, ncols = size(mask)
    seen = falses(nrows, ncols)
    sizes = Int[]
    qj = Int[]
    qi = Int[]
    for j in 1:nrows, i in 1:ncols
        (mask[j, i] && !seen[j, i]) || continue
        push!(sizes, 0)
        empty!(qj); empty!(qi)
        push!(qj, j); push!(qi, i); seen[j, i] = true
        head = 1
        while head <= length(qj)
            cj = qj[head]; ci = qi[head]; head += 1
            sizes[end] += 1
            nb = ((mod1(cj - 1, nrows), ci), (mod1(cj + 1, nrows), ci),
                  (cj, mod1(ci - 1, ncols)), (cj, mod1(ci + 1, ncols)))
            for (nj, ni) in nb
                if mask[nj, ni] && !seen[nj, ni]
                    seen[nj, ni] = true
                    push!(qj, nj); push!(qi, ni)
                end
            end
        end
    end
    total_active = sum(sizes)
    return (
        components = length(sizes),
        active_cells = total_active,
        largest_component_cells = isempty(sizes) ? 0 : maximum(sizes),
        largest_component_fraction_of_active = total_active == 0 ? 0.0 : maximum(sizes) / total_active,
    )
end

function _fit_summary(est, shape_mask::AbstractMatrix{Bool}, target_mask::AbstractMatrix{Bool})
    total = grid_density_mass(est)
    inter = shape_mask .& target_mask
    union = shape_mask .| target_mask
    shape_mass = grid_density_mass(est, shape_mask)
    target_mass = grid_density_mass(est, target_mask)
    intersection_mass = grid_density_mass(est, inter)
    union_mass = grid_density_mass(est, union)
    outside_mass = max(shape_mass - intersection_mass, 0.0)

    shape_area = count(shape_mask)
    target_area = count(target_mask)
    intersection_area = count(inter)
    union_area = count(union)
    outside_area = max(shape_area - intersection_area, 0)
    ncells = length(shape_mask)
    mass_jaccard = union_mass <= 0 ? 0.0 : intersection_mass / union_mass
    area_jaccard = union_area == 0 ? 0.0 : intersection_area / union_area
    fit_score = 0.5 * mass_jaccard + 0.5 * area_jaccard
    return (
        shape_mass_fraction = total <= 0 ? 0.0 : shape_mass / total,
        target_mass_fraction = total <= 0 ? 0.0 : target_mass / total,
        target_coverage = target_mass <= 0 ? 0.0 : intersection_mass / target_mass,
        outside_leakage = total <= 0 ? 0.0 : outside_mass / total,
        leakage_fraction_of_shape = shape_mass <= 0 ? 0.0 : outside_mass / shape_mass,
        area_fraction = shape_area / ncells,
        target_area_fraction = target_area / ncells,
        area_coverage = target_area == 0 ? 0.0 : intersection_area / target_area,
        outside_area_fraction = outside_area / ncells,
        mass_jaccard = mass_jaccard,
        area_jaccard = area_jaccard,
        fit_score = fit_score,
    )
end

function alpha_candidate_values(triangles::AbstractVector{AlphaTriangle},
                                density_gate::Real;
                                nsamples::Int=200,
                                linear_fraction::Real=0.5)
    vals = sort(unique(t.grade for t in triangles if t.gate_density >= Float64(density_gate)))
    ns = Int(nsamples)
    ns >= 2 || throw(ArgumentError("nsamples must be at least 2."))
    isempty(vals) && return Float64[]
    length(vals) <= ns && return vals

    frac = clamp(Float64(linear_fraction), 0.0, 1.0)
    nlinear = clamp(round(Int, frac * ns), 2, ns)
    nquantile = max(ns - nlinear, 2)

    quantile_grid = range(0.0, 1.0; length=nquantile)
    quantile_vals = Float64[quantile(vals, q) for q in quantile_grid]
    linear_vals = collect(range(first(vals), last(vals); length=nlinear))

    candidates = sort(unique(vcat(quantile_vals, linear_vals, first(vals), last(vals))))
    return candidates
end

function alpha_mask_and_fit(est,
                            triangles::AbstractVector{AlphaTriangle},
                            alpha::Real,
                            density_gate::Real,
                            target_mask::AbstractMatrix{Bool};
                            cache=Dict{Tuple{Float64,Float64},Tuple{BitMatrix,NamedTuple}}())
    key = (Float64(alpha), Float64(density_gate))
    if haskey(cache, key)
        return cache[key]
    end
    mask = rasterize_alpha_mask(triangles, key[1], key[2], est.phi_grid, est.psi_grid)
    summary = _fit_summary(est, mask, target_mask)
    cache[key] = (mask, summary)
    return mask, summary
end

const ALPHA_PARSIMONY_FIT_FRACTION = 0.99

function choose_alpha_for_density_gate(est,
                                       triangles::AbstractVector{AlphaTriangle};
                                       target::Real,
                                       density_gate::Real,
                                       nsamples::Int=200,
                                       parsimony_fraction::Real=ALPHA_PARSIMONY_FIT_FRACTION)
    gate = Float64(density_gate)
    target_mask = (hasproperty(est, :point_values) ? est.point_values : est.density) .>= gate
    candidates = alpha_candidate_values(triangles, gate; nsamples=nsamples)
    isempty(candidates) && error("density gate selected no eligible alpha triangles")

    parsimony = Float64(parsimony_fraction)
    0 < parsimony <= 1 || throw(ArgumentError("parsimony_fraction must lie in (0, 1]."))

    births = _alpha_birth_grid(triangles, gate, est.phi_grid, est.psi_grid)
    rows = NamedTuple[]
    best_idx = 1
    best_score = -Inf
    best_alpha = Inf
    for alpha in candidates
        mask = births .<= alpha
        fit = _fit_summary(est, mask, target_mask)
        comp = mask_component_summary_torus(mask)
        row = (; alpha=Float64(alpha), fit..., comp...)
        push!(rows, row)
        score = row.fit_score
        if score > best_score + 1.0e-12 || (abs(score - best_score) <= 1.0e-12 && alpha < best_alpha)
            best_score = score
            best_alpha = alpha
            best_idx = length(rows)
        end
    end
    maxfit = rows[best_idx]
    threshold = parsimony * maxfit.fit_score
    parsimony_idx = findfirst(row -> row.fit_score >= threshold, rows)
    parsimony_idx === nothing && (parsimony_idx = best_idx)
    parsimonious = rows[parsimony_idx]

    maxfit_mask = births .<= maxfit.alpha
    parsimonious_mask = births .<= parsimonious.alpha
    return (; target=Float64(target), density_gate=gate,
        alpha=parsimonious.alpha,
        mask=parsimonious_mask,
        target_mask,
        selection_rule=:parsimonious_within_fraction,
        parsimony_fraction=parsimony,
        maxfit_alpha=maxfit.alpha,
        maxfit_score=maxfit.fit_score,
        maxfit_mask,
        parsimonious_alpha=parsimonious.alpha,
        parsimonious_score=parsimonious.fit_score,
        parsimonious..., sweep=rows)
end


include("rama_alpha_contours/integration.jl")

# At a fixed density gate, a sample enters the alpha support at the smallest
# radius of an eligible triangle containing it. One geometry traversal therefore
# gives every candidate mask, without changing the alpha-selection rule.
function _alpha_birth_grid(triangles, density_gate, phi_grid, psi_grid)
    birth = fill(Inf, length(psi_grid), length(phi_grid))
    xlimits, ylimits = (first(phi_grid), last(phi_grid)), (first(psi_grid), last(psi_grid))
    for tri in triangles
        tri.gate_density >= density_gate || continue
        foreach_periodic_triangle(tri; xlimits=xlimits, ylimits=ylimits) do a, b, c
            ir = _grid_index_range(phi_grid, min(a[1],b[1],c[1]), max(a[1],b[1],c[1]))
            jr = _grid_index_range(psi_grid, min(a[2],b[2],c[2]), max(a[2],b[2],c[2]))
            for j in jr, i in ir
                tri.grade < birth[j,i] || continue
                _point_in_triangle(Float64(phi_grid[i]), Float64(psi_grid[j]), a, b, c) || continue
                birth[j,i] = tri.grade
            end
        end
    end
    return birth
end

"""Sample the filled alpha support on a periodic grid; isolated vertices are omitted."""
function rasterize_alpha_mask(triangles::AbstractVector{AlphaTriangle}, alpha::Real,
        density_gate::Real, phi_grid::AbstractVector{<:Real}, psi_grid::AbstractVector{<:Real})
    alpha >= 0 || throw(ArgumentError("alpha must be nonnegative"))
    isfinite(density_gate) || throw(ArgumentError("density_gate must be finite"))
    mask = falses(length(psi_grid), length(phi_grid))
    xlimits, ylimits = (first(phi_grid), last(phi_grid)), (first(psi_grid), last(psi_grid))
    for tri in triangles
        tri.grade <= alpha && tri.gate_density >= density_gate || continue
        foreach_periodic_triangle(tri; xlimits=xlimits, ylimits=ylimits) do a, b, c
            ir = _grid_index_range(phi_grid, min(a[1], b[1], c[1]), max(a[1], b[1], c[1]))
            jr = _grid_index_range(psi_grid, min(a[2], b[2], c[2]), max(a[2], b[2], c[2]))
            for j in jr, i in ir
                mask[j, i] && continue
                mask[j, i] = _point_in_triangle(Float64(phi_grid[i]), Float64(psi_grid[j]), a, b, c)
            end
        end
    end
    return mask
end

include("rama_alpha_contours/artifacts.jl")

"""
Fraction of weighted observations inside retained triangles, including their edges.
This measures the geometric region itself, not the display mask or vertex list.
It is in-sample coverage when `points` are the construction observations.
"""
function empirical_support_fraction(points::AbstractMatrix{<:Real},
        weights::AbstractVector{<:Real}, triangles::AbstractVector{AlphaTriangle},
        alpha::Real, density_gate::Real; bucket_grid::Int=64)
    _check_points(points)
    length(weights) == size(points, 1) || throw(DimensionMismatch("one weight per point is required"))
    all(x -> isfinite(x) && x >= 0, weights) && sum(weights) > 0 || throw(ArgumentError("weights must be finite, nonnegative, and have positive sum"))
    bucket_grid > 0 || throw(ArgumentError("bucket_grid must be positive"))
    alpha >= 0 && isfinite(density_gate) || throw(ArgumentError("invalid alpha or density gate"))
    n = size(points, 1)
    xs = [_wrap_angle(points[i, 1]) for i in 1:n]
    ys = [_wrap_angle(points[i, 2]) for i in 1:n]
    heads = zeros(Int, bucket_grid, bucket_grid)
    next = zeros(Int, n)
    for p in 1:n
        i, j = _grid_bin_index(xs[p], bucket_grid), _grid_bin_index(ys[p], bucket_grid)
        next[p] = heads[j, i]
        heads[j, i] = p
    end
    inside = falses(n)
    for tri in triangles
        tri.grade <= alpha && tri.gate_density >= density_gate || continue
        foreach_periodic_triangle(tri) do a, b, c
            minx = max(DOMAIN_MIN, min(a[1], b[1], c[1]))
            maxx = min(DOMAIN_MAX, max(a[1], b[1], c[1]))
            miny = max(DOMAIN_MIN, min(a[2], b[2], c[2]))
            maxy = min(DOMAIN_MAX, max(a[2], b[2], c[2]))
            minx <= maxx && miny <= maxy || return
            i0 = clamp(floor(Int, (minx - DOMAIN_MIN) / PERIOD * bucket_grid) + 1, 1, bucket_grid)
            i1 = clamp(floor(Int, (maxx - DOMAIN_MIN) / PERIOD * bucket_grid) + 1, 1, bucket_grid)
            j0 = clamp(floor(Int, (miny - DOMAIN_MIN) / PERIOD * bucket_grid) + 1, 1, bucket_grid)
            j1 = clamp(floor(Int, (maxy - DOMAIN_MIN) / PERIOD * bucket_grid) + 1, 1, bucket_grid)
            for j in j0:j1, i in i0:i1
                p = heads[j, i]
                while p != 0
                    if !inside[p] && _point_in_triangle(xs[p], ys[p], a, b, c)
                        inside[p] = true
                    end
                    p = next[p]
                end
            end
        end
    end
    return sum(weights[inside]) / sum(weights)
end

# Occupied raster samples are interpreted as closed periodic square cells here.
# Counting shared vertices/edges avoids ambiguous claims about continuum topology.
function raster_topology(mask::AbstractMatrix{Bool})
    nr, nc = size(mask)
    vertices = falses(nr, nc)
    horizontal = falses(nr, nc)
    vertical = falses(nr, nc)
    for j in 1:nr, i in 1:nc
        mask[j, i] || continue
        jp, ip = mod1(j + 1, nr), mod1(i + 1, nc)
        vertices[j, i] = vertices[jp, i] = vertices[j, ip] = vertices[jp, ip] = true
        horizontal[j, i] = horizontal[jp, i] = true
        vertical[j, i] = vertical[j, ip] = true
    end
    # Closed cells meeting at a corner are connected: use eight neighbors.
    seen = falses(nr, nc)
    queue = CartesianIndex{2}[]
    components = 0
    for p in CartesianIndices(mask)
        mask[p] && !seen[p] || continue
        components += 1
        empty!(queue); push!(queue, p); seen[p] = true
        k = 1
        while k <= length(queue)
            j, i = Tuple(queue[k]); k += 1
            for dj in -1:1, di in -1:1
                dj == 0 && di == 0 && continue
                q = CartesianIndex(mod1(j + dj, nr), mod1(i + di, nc))
                if mask[q] && !seen[q]
                    seen[q] = true; push!(queue, q)
                end
            end
        end
    end
    euler = count(vertices) - count(horizontal) - count(vertical) + count(mask)
    b2 = all(mask) ? 1 : 0
    return (components=components, loops=components + b2 - euler, euler=euler)
end

function _raster_boundary_points(mask, phi_grid, psi_grid)
    nr, nc = size(mask)
    dx, dy = _grid_step(phi_grid), _grid_step(psi_grid)
    points = NTuple{2,Float64}[]
    for j in 1:nr, i in 1:nc
        mask[j, i] || continue
        mask[j, mod1(i + 1, nc)] || push!(points, (_wrap_angle(phi_grid[i] + dx / 2), Float64(psi_grid[j])))
        mask[j, mod1(i - 1, nc)] || push!(points, (_wrap_angle(phi_grid[i] - dx / 2), Float64(psi_grid[j])))
        mask[mod1(j + 1, nr), i] || push!(points, (Float64(phi_grid[i]), _wrap_angle(psi_grid[j] + dy / 2)))
        mask[mod1(j - 1, nr), i] || push!(points, (Float64(phi_grid[i]), _wrap_angle(psi_grid[j] - dy / 2)))
    end
    return points
end

function _directed_boundary_distance(a, b)
    result = 0.0
    for p in a
        best = Inf
        for q in b
            dx, dy = abs(p[1] - q[1]), abs(p[2] - q[2])
            best = min(best, hypot(min(dx, PERIOD - dx), min(dy, PERIOD - dy)))
        end
        result = max(result, best)
    end
    return result
end

"""Largest nearest-boundary displacement between sampled periodic boundaries, in degrees."""
function raster_boundary_distance(a, b, phi_grid, psi_grid)
    size(a) == size(b) == (length(psi_grid), length(phi_grid)) || throw(DimensionMismatch("boundary masks and grids differ"))
    a == b && return 0.0
    pa, pb = _raster_boundary_points(a, phi_grid, psi_grid), _raster_boundary_points(b, phi_grid, psi_grid)
    (isempty(pa) || isempty(pb)) && return Inf
    return max(_directed_boundary_distance(pa, pb), _directed_boundary_distance(pb, pa))
end

"""
Compare two refitted alpha supports using one common reference density and grid.
Tolerances are engineering acceptance criteria for this dataset, not confidence bounds.
"""
function support_comparison(analysis, reference; points, weights,
        coverage_tolerance::Real=0.001, area_tolerance::Real=0.01,
        boundary_tolerance_deg::Real=PERIOD / length(reference.del_density.phi_grid))
    for (name, value) in ((:coverage_tolerance, coverage_tolerance),
            (:area_tolerance, area_tolerance), (:boundary_tolerance_deg, boundary_tolerance_deg))
        isfinite(value) && value >= 0 || throw(ArgumentError("$(name) must be finite and nonnegative"))
    end
    length(analysis.alpha_results) == length(reference.alpha_results) || throw(ArgumentError("comparison target counts must match"))
    est, ref = analysis.del_density, reference.del_density
    est.phi_grid == ref.phi_grid && est.psi_grid == ref.psi_grid || throw(ArgumentError("comparison needs the same grid"))
    mass, refmass = grid_density_mass(est), grid_density_mass(ref)
    density_l1 = sum(abs.(est.density ./ mass .- ref.density ./ refmass)) * grid_cell_area(ref)
    rows = NamedTuple[]
    for (r, s) in zip(analysis.alpha_results, reference.alpha_results)
        r.target == s.target || throw(ArgumentError("comparison targets must match"))
        coverage = empirical_support_fraction(points, weights, analysis.alpha_triangles, r.alpha, r.density_gate)
        refcoverage = empirical_support_fraction(points, weights, reference.alpha_triangles, s.alpha, s.density_gate)
        area, refarea = count(r.mask), count(s.mask)
        relative_area_change = refarea == 0 ? (area == 0 ? 0.0 : Inf) : abs(area - refarea) / refarea
        union_count = count(r.mask .| s.mask)
        jaccard = union_count == 0 ? 1.0 : count(r.mask .& s.mask) / union_count
        boundary = raster_boundary_distance(r.mask, s.mask, ref.phi_grid, ref.psi_grid)
        topology, reftopology = raster_topology(r.mask), raster_topology(s.mask)
        passed = abs(coverage - refcoverage) <= coverage_tolerance &&
            relative_area_change <= area_tolerance && boundary <= boundary_tolerance_deg &&
            topology == reftopology
        push!(rows, (target=r.target, alpha=r.alpha, reference_alpha=s.alpha,
            empirical_coverage=coverage, reference_empirical_coverage=refcoverage,
            coverage_change=coverage - refcoverage,
            reference_density_mass=grid_density_mass(ref, r.mask) / refmass,
            reference_support_density_mass=grid_density_mass(ref, s.mask) / refmass,
            density_normalized_l1=density_l1, mask_jaccard=jaccard,
            relative_area_change=relative_area_change, boundary_displacement_deg=boundary,
            raster_components=topology.components, reference_raster_components=reftopology.components,
            raster_loops=topology.loops, reference_raster_loops=reftopology.loops,
            agreement_within_tolerances=passed))
    end
    return rows
end

"""
    ghost_protocol_comparison(points; category, ...)

Compare full, adaptive, and experimental fixed periodic copying on identical
observations. Refit each density gate and alpha radius before comparing filled
regions. Coverage is evaluated at the supplied observations, including when
triangulation uses weighted grid representatives.

Geometry timings use fresh calculations after a small-kernel warmup, with the
full reference first in every pass. Artifact construction and cache retrieval
are reported separately. The operational 5% time-reduction threshold does not
supply a statistical confidence interval or authorize fixed copying elsewhere.
"""
function ghost_protocol_comparison(points::AbstractMatrix{<:Real};
        category::AbstractString, data_path::AbstractString="", grid_size::Int=96,
        ghost_margin::Real=45.0, seed::Integer=2026,
        grid_origin=(-180.0,-180.0), area_floor_quantile::Real=0.01,
        use_grid_representatives::Bool=false, representative_grid::Int=256,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        sigma_deg::Real=8.0, mass_targets=(0.98, 0.995), nsamples::Int=200,
        parsimony_fraction::Real=0.99, repeats::Int=2,
        coverage_tolerance::Real=0.001, area_tolerance::Real=0.01,
        boundary_tolerance_deg::Real=PERIOD / grid_size, progress=_quiet_progress)
    _check_points(points)
    grid_size >= 2 || throw(ArgumentError("grid_size must be at least 2"))
    periodic_density_grid(; grid_size, grid_origin)
    isfinite(area_floor_quantile) && 0 <= area_floor_quantile <= 1 ||
        throw(ArgumentError("area_floor_quantile must lie in [0, 1]"))
    repeats >= 2 || throw(ArgumentError("use at least two independent geometry timing passes"))
    isfinite(ghost_margin) && 0 <= ghost_margin <= PERIOD ||
        throw(ArgumentError("ghost_margin must lie in [0, 360]"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    isfinite(sigma_deg) && sigma_deg >= 0 ||
        throw(ArgumentError("sigma_deg must be finite and nonnegative"))
    nsamples >= 2 || throw(ArgumentError("nsamples must be at least 2"))
    0 < parsimony_fraction <= 1 ||
        throw(ArgumentError("parsimony_fraction must lie in (0, 1]"))
    targets = Tuple(Float64(q) for q in mass_targets)
    !isempty(targets) && all(q -> isfinite(q) && 0 < q <= 1, targets) ||
        throw(ArgumentError("mass_targets must be nonempty fractions in (0, 1]"))
    for (name, value) in ((:coverage_tolerance, coverage_tolerance),
            (:area_tolerance, area_tolerance), (:boundary_tolerance_deg, boundary_tolerance_deg))
        isfinite(value) && value >= 0 || throw(ArgumentError("$(name) must be finite and nonnegative"))
    end

    modes = (:full, :adaptive, :fixed)
    input_points, _, input_summary = delaunay_input_points_with_counts(points;
        use_grid_representatives=use_grid_representatives,
        representative_grid=representative_grid, progress=progress)
    # Use the same matrix and progress-callback types for warmup and timing.
    # The small fixture warms the algorithm, not the production artifact cache.
    warmup = PERIOD .* rand(Random.Xoshiro(seed), 80, 2) .+ DOMAIN_MIN
    progress("warming full, adaptive, and fixed geometry on an independent 80-point fixture")
    for mode in modes
        periodic_delaunay_density_triangles(warmup; mode=mode, margin=ghost_margin,
            seed=seed, progress=_quiet_progress)
    end
    timing_rows = NamedTuple[]
    geometry_summaries = Dict{Symbol,NamedTuple}()
    for pass in 1:repeats
        for mode in modes
            progress("geometry timing pass $(pass)/$(repeats): $(mode), $(size(input_points, 1)) input points")
            GC.gc()
            measured = @timed begin
                _, _, summary = periodic_delaunay_density_triangles(input_points;
                    mode=mode, margin=ghost_margin, seed=seed, progress=_quiet_progress)
                # Do not retain another complete triangulation for every pass.
                summary
            end
            summary = measured.value
            geometry_summaries[mode] = summary
            push!(timing_rows, (mode=mode, pass=pass,
                timing_policy=:warm_uncached_geometry, seconds=measured.time,
                allocated_bytes=measured.bytes, gc_seconds=measured.gctime,
                compile_seconds=hasproperty(measured, :compile_time) ? measured.compile_time : nothing,
                n_input_points=size(input_points, 1), n_tiled_points=summary.n_tiled_points,
                n_triangles=summary.n_triangles, certified=summary.certified,
                accepted_margin=summary.margin, attempt_count=length(summary.attempts),
                attempted_tiled_points=sum(a.n_tiled_points for a in summary.attempts),
                attempts=summary.attempts))
        end
    end

    analyses = Dict{Symbol,NamedTuple}()
    artifact_timing_rows = NamedTuple[]
    failures = Dict{Symbol,String}()
    for mode in modes
        if mode == :fixed && geometry_summaries[mode].n_triangles == 0
            failures[mode] = "fixed margin produced no triangles; no density or contour comparison is possible"
            continue
        end
        progress("constructing or loading $(mode) density artifact and refitting its alpha supports")
        artifact_measurement = try
            @timed delaunay_density_artifact(points;
                category=category, data_path=data_path, grid_size=grid_size,
                grid_origin=grid_origin, area_floor_quantile=area_floor_quantile,
                geometry_mode=mode, ghost_margin=ghost_margin, seed=seed,
                use_grid_representatives=use_grid_representatives,
                representative_grid=representative_grid, cache_dir=cache_dir,
                refresh_cache=refresh_cache, progress=progress)
        catch err
            # This is an expected scientific outcome of an incomplete fixed
            # margin. Do not hide I/O errors, invalid settings, or code defects.
            if mode == :fixed && err isa ErrorException &&
                    err.msg in ("periodic Delaunay produced no positive point areas", "periodic Delaunay produced zero point areas")
                failures[mode] = err.msg
                continue
            end
            rethrow()
        end
        artifact = artifact_measurement.value
        push!(artifact_timing_rows, (mode=mode, operation=:artifact_load_or_build,
            seconds=artifact_measurement.time, allocated_bytes=artifact_measurement.bytes,
            gc_seconds=artifact_measurement.gctime, cache_status=artifact.cache_status,
            cache_file=artifact.cache_file))
        if cache_dir !== nothing
            # Explicitly time a subsequent retrieval instead of describing an
            # initial cache miss/build as a cached timing. Discard its geometry.
            cached_measurement = @timed begin
                cached = delaunay_density_artifact(points;
                    category=category, data_path=data_path, grid_size=grid_size,
                grid_origin=grid_origin, area_floor_quantile=area_floor_quantile,
                    geometry_mode=mode, ghost_margin=ghost_margin, seed=seed,
                    use_grid_representatives=use_grid_representatives,
                    representative_grid=representative_grid, cache_dir=cache_dir,
                    refresh_cache=false, progress=_quiet_progress)
                (cache_status=cached.cache_status, cache_file=cached.cache_file)
            end
            push!(artifact_timing_rows, (mode=mode, operation=:subsequent_artifact_retrieval,
                seconds=cached_measurement.time, allocated_bytes=cached_measurement.bytes,
                gc_seconds=cached_measurement.gctime,
                cache_status=cached_measurement.value.cache_status,
                cache_file=cached_measurement.value.cache_file))
        end
        analysis = try
            alpha_analysis(artifact; sigma_deg=sigma_deg, mass_targets=targets,
                nsamples=nsamples, parsimony_fraction=parsimony_fraction)
        catch err
            if mode == :fixed && err isa ErrorException && err.msg in
                    ("density field has zero mass", "density gate selected no eligible alpha triangles",
                     "periodic density does not cover every cell exactly once")
                failures[mode] = err.msg
                continue
            end
            rethrow()
        end
        analyses[mode] = analysis
    end

    reference = analyses[:full]
    # These are the actual supplied observations, not the representative grid.
    # Repeated observations naturally contribute their full multiplicity.
    observation_weights = ones(Int, size(points, 1))
    comparison_rows = NamedTuple[]
    comparisons_by_mode = Dict{Symbol,Vector{NamedTuple}}()
    for mode in modes
        haskey(analyses, mode) || continue
        rows = support_comparison(analyses[mode], reference;
            points=points, weights=observation_weights,
            coverage_tolerance=coverage_tolerance, area_tolerance=area_tolerance,
            boundary_tolerance_deg=boundary_tolerance_deg)
        comparisons_by_mode[mode] = rows
        append!(comparison_rows, [merge((mode=mode,), row) for row in rows])
    end

    reference_seconds = median(row.seconds for row in timing_rows if row.mode == :full)
    reference_bytes = median(row.allocated_bytes for row in timing_rows if row.mode == :full)
    assessment = Dict{Symbol,NamedTuple}()
    for mode in modes
        mode_timings = [row for row in timing_rows if row.mode == mode]
        seconds = median(row.seconds for row in mode_timings)
        bytes = median(row.allocated_bytes for row in mode_timings)
        reduction = 1.0 - seconds / reference_seconds
        speed_advantage = mode != :full && reduction >= 0.05
        usable = haskey(comparisons_by_mode, mode)
        rows = get(comparisons_by_mode, mode, NamedTuple[])
        agreement = usable && all(row.agreement_within_tolerances for row in rows)
        status = mode == :full ? :reference : !usable ? :unusable :
            !agreement ? :outside_tolerances : speed_advantage ?
            :supported_on_this_dataset : :agreement_without_speed_advantage
        assessment[mode] = (status=status,
            experimental_approximation=mode == :fixed,
            certified=geometry_summaries[mode].certified,
            agreement_within_tolerances=agreement,
            measured_speed_advantage=speed_advantage,
            median_geometry_seconds=seconds,
            median_allocated_bytes=bytes,
            median_gc_seconds=median(row.gc_seconds for row in mode_timings),
            geometry_speedup=reference_seconds / seconds,
            geometry_time_reduction_fraction=reduction,
            allocated_bytes_change=bytes - reference_bytes,
            max_absolute_coverage_change=usable ? maximum(abs(row.coverage_change) for row in rows) : nothing,
            normalized_density_l1=usable ? first(rows).density_normalized_l1 : nothing,
            max_relative_area_change=usable ? maximum(row.relative_area_change for row in rows) : nothing,
            max_boundary_displacement_deg=usable ? maximum(row.boundary_displacement_deg for row in rows) : nothing,
            topology_unchanged=usable && all(row.raster_components == row.reference_raster_components &&
                row.raster_loops == row.reference_raster_loops for row in rows),
            failure_reason=get(failures, mode, nothing),
            scope=:provided_dataset_and_settings_only,
            production_fixed_mode_approval=false)
    end

    metadata = (category=String(category),
        data_path=isempty(data_path) ? "" : abspath(data_path),
        point_hash=_matrix_sha1(points), source_sha1=_source_sha1(),
        n_observations=size(points, 1), n_input_points=size(input_points, 1),
        representative_summary=input_summary,
        input_scope=:provided_points,
        limits=(upstream_row_limit=:unspecified, upstream_reference_point_limit=:unspecified),
        julia_version=string(VERSION),
        triangulation_version=string(Base.pkgversion(RamaAlphaGeometry.DT)),
        timing_policy=:warm_uncached_geometry, timing_order=modes,
        timing_repeats=repeats, warmup_points=80,
        allocation_measure=:total_allocated_bytes_not_peak_memory,
        cache_timing_policy=:separately_reported_artifact_calls,
        settings=(grid_size=grid_size, ghost_margin=Float64(ghost_margin), seed=Int(seed),
            use_grid_representatives=use_grid_representatives,
            representative_grid=use_grid_representatives ? representative_grid : nothing,
            sigma_deg=Float64(sigma_deg), mass_targets=targets, nsamples=nsamples,
            parsimony_fraction=Float64(parsimony_fraction), area_floor_quantile=Float64(area_floor_quantile),
            grid_origin=periodic_density_grid(;grid_size=grid_size,grid_origin=grid_origin).grid_origin,
            integration_method=:triangle_cell_exact),
        tolerances=(coverage=Float64(coverage_tolerance), relative_area=Float64(area_tolerance),
            boundary_deg=Float64(boundary_tolerance_deg), unchanged_raster_topology=true,
            minimum_geometry_time_reduction=0.05),
        tolerance_interpretation=:operational_not_biologically_or_statistically_validated,
        timing_uncertainty=:not_estimated,
        empirical_coverage_scope=:construction_observations_in_filled_regions,
        fixed_mode_scope=:comparison_experiment_only)
    return (; analyses, comparison_rows, timing_rows, artifact_timing_rows,
        assessment, geometry_summaries, metadata)
end

include("rama_alpha_contours/studies.jl")

end # module RamaAlphaContours
