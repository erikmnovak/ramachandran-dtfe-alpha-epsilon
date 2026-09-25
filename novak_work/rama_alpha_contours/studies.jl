# Numerical studies use the same certified triangulation for every estimator.
# A change of grid must not also change the points or the Delaunay geometry.

function _check_study_geometry(geometry, points)
    _check_points(points)
    geometry.tiling_summary.certified ||
        throw(ArgumentError("density studies require certified periodic geometry"))
    return nothing
end

function _check_study_settings(; grid_sizes, shift_fractions, sigma_values,
        area_floor_quantiles, mass_targets, nsamples, audit_grid_size)
    !isempty(grid_sizes) && all(n -> n isa Integer && n >= 2, grid_sizes) ||
        throw(ArgumentError("grid sizes must be integers of at least 2"))
    !isempty(shift_fractions) && all(x -> isfinite(x) && 0 <= x < 1, shift_fractions) ||
        throw(ArgumentError("origin shifts must be fractions in [0, 1)"))
    0 in shift_fractions || throw(ArgumentError("origin shifts must include zero"))
    !isempty(sigma_values) && all(x -> isfinite(x) && x >= 0, sigma_values) ||
        throw(ArgumentError("smoothing scales must be finite and nonnegative"))
    !isempty(area_floor_quantiles) && all(x -> isfinite(x) && 0 <= x <= 1, area_floor_quantiles) ||
        throw(ArgumentError("area-floor quantiles must lie in [0, 1]"))
    !isempty(mass_targets) && all(x -> isfinite(x) && 0 < x <= 1, mass_targets) ||
        throw(ArgumentError("mass targets must lie in (0, 1]"))
    nsamples >= 2 || throw(ArgumentError("nsamples must be at least 2"))
    audit_grid_size >= 2 || throw(ArgumentError("audit_grid_size must be at least 2"))
    return nothing
end

# For the observations that built an exact (uncompressed) triangulation, a
# point belongs to a filled triangle union iff some retained triangle is
# incident to that vertex. This is an exact containment test, not the tempting
# but incorrect test that all alpha-complex vertices count as filled support.
function _study_observations(geometry, points)
    point_hash = _matrix_sha1(points)
    construction_observations = point_hash == geometry.metadata.point_hash
    exact_vertices = !geometry.metadata.use_grid_representatives &&
        construction_observations
    return (points=exact_vertices ? geometry.unique_pts : points,
        weights=exact_vertices ? geometry.multiplicities : ones(Int, size(points, 1)),
        vertex_incidence=exact_vertices, point_hash=point_hash,
        coverage_kind=construction_observations ? :in_sample_geometric_containment :
            :supplied_point_geometric_containment,
        evaluation=exact_vertices ? :construction_vertex_incidence : :geometric_point_queries)
end

function _study_support_coverage(analysis, observations, alpha, gate)
    if !observations.vertex_incidence
        return empirical_support_fraction(observations.points, observations.weights,
            analysis.alpha_triangles, alpha, gate)
    end
    length(analysis.density_triangles) == length(analysis.alpha_triangles) ||
        error("alpha triangles must retain the density-triangle indexing")
    inside = falses(length(observations.weights))
    for (tri, alpha_tri) in zip(analysis.density_triangles, analysis.alpha_triangles)
        alpha_tri.grade <= alpha && alpha_tri.gate_density >= gate || continue
        for vertex in tri.base
            inside[vertex] = true
        end
    end
    return sum(observations.weights[i] for i in eachindex(inside) if inside[i]; init=0) /
        sum(observations.weights)
end

function _study_raw_coverage(artifact, observations, gate)
    if observations.vertex_incidence
        return sum(observations.weights[i] for i in eachindex(observations.weights)
            if artifact.point_density[i] >= gate; init=0) / sum(observations.weights)
    end
    return empirical_superlevel_fraction(observations.points, observations.weights,
        artifact.density_triangles, artifact.point_density, gate)
end

function _study_tolerances(; density_l1_tolerance, coverage_tolerance,
        area_tolerance, boundary_tolerance_deg, gate_relative_tolerance,
        alpha_relative_tolerance, audit_area_tolerance,
        audit_boundary_tolerance_deg, audit_density_l1_tolerance)
    result = (; density_l1_tolerance, coverage_tolerance, area_tolerance,
        boundary_tolerance_deg, gate_relative_tolerance, alpha_relative_tolerance,
        audit_area_tolerance, audit_boundary_tolerance_deg, audit_density_l1_tolerance)
    all(x -> isfinite(x) && x >= 0, values(result)) ||
        throw(ArgumentError("study tolerances must be finite and nonnegative"))
    return result
end

_relative_change(value, reference) = reference == 0 ? (value == 0 ? 0.0 : Inf) :
    abs(Float64(value) - Float64(reference)) / abs(Float64(reference))

function _exact_density_mass(triangles, point_density)
    return sum(t.area * (point_density[t.base[1]] + point_density[t.base[2]] +
        point_density[t.base[3]]) / 3 for t in triangles)
end

function _study_floor_summary(artifact, quantile_value)
    observed_mass = Float64(sum(artifact.multiplicities))
    estimator_mass = _exact_density_mass(artifact.density_triangles, artifact.point_density)
    positive_areas = filter(>(0), artifact.point_area)
    floor_area = quantile_value == 0 ? 0.0 : quantile(positive_areas, quantile_value)
    affected = count(x -> x < floor_area, artifact.point_area)
    return (area_floor_quantile=Float64(quantile_value), floor_area_deg2=floor_area,
        n_vertices=length(artifact.point_area), n_vertices_affected=affected,
        fraction_vertices_affected=affected / length(artifact.point_area),
        observation_weight=observed_mass, estimator_mass=estimator_mass,
        mass_removed_by_floor=observed_mass - estimator_mass,
        fraction_mass_removed_by_floor=(observed_mass - estimator_mass) / observed_mass)
end

function _study_density_values(analysis, phi_grid, psi_grid)
    if analysis.del_density.smoothing_sigma_deg == 0
        density, covered = sample_triangle_density(analysis.density_triangles,
            analysis.point_density, phi_grid, psi_grid)
        all(covered) || error("certified geometry left an uncovered density audit point")
        return density
    end
    return [_periodic_bilinear_value(analysis.del_density, x, y)
        for y in psi_grid, x in phi_grid]
end

# Exact squared Euclidean distance on the doubled periodic pixel lattice. An
# edge midpoint of an audit cell is a lattice point. This measures the same
# midpoint sets as raster_boundary_distance, without its quadratic pair scan.
function _periodic_distance_line!(out, input, n, sites, breaks)
    m = 3n
    k = 0
    for q in 1:m
        fq = input[mod1(q, n)]
        isfinite(fq) || continue
        s = -Inf
        while k > 0
            p = sites[k]
            s = ((fq + q * q) - (input[mod1(p, n)] + p * p)) / (2 * (q - p))
            s > breaks[k] && break
            k -= 1
        end
        k += 1
        sites[k] = q
        breaks[k] = k == 1 ? -Inf : s
        breaks[k + 1] = Inf
    end
    if k == 0
        fill!(out, Inf)
        return out
    end
    envelope = 1
    for q in (n + 1):(2n)
        while breaks[envelope + 1] < q
            envelope += 1
        end
        p = sites[envelope]
        out[q - n] = (q - p)^2 + input[mod1(p, n)]
    end
    return out
end

function _boundary_lattice(mask)
    nr, nc = size(mask)
    boundary = falses(2nr, 2nc)
    for j in 1:nr, i in 1:nc
        mask[j, i] || continue
        mask[j, mod1(i + 1, nc)] || (boundary[2j - 1, 2i] = true)
        mask[j, mod1(i - 1, nc)] || (boundary[2j - 1, mod1(2i - 2, 2nc)] = true)
        mask[mod1(j + 1, nr), i] || (boundary[2j, 2i - 1] = true)
        mask[mod1(j - 1, nr), i] || (boundary[mod1(2j - 2, 2nr), 2i - 1] = true)
    end
    return boundary
end

function _boundary_distance_field!(work, boundary)
    nr, nc = size(boundary)
    nr == nc || throw(ArgumentError("the study boundary audit requires a square grid"))
    line = Vector{Float64}(undef, nr)
    output = similar(line)
    sites = Vector{Int}(undef, 3nr)
    breaks = Vector{Float64}(undef, 3nr + 1)
    for j in 1:nr
        for i in 1:nc
            line[i] = boundary[j, i] ? 0.0 : Inf
        end
        _periodic_distance_line!(output, line, nc, sites, breaks)
        for i in 1:nc
            work[j, i] = output[i]
        end
    end
    for i in 1:nc
        copyto!(line, view(work, :, i))
        _periodic_distance_line!(output, line, nr, sites, breaks)
        copyto!(view(work, :, i), output)
    end
    return work
end

function _study_boundary_distance(a, b)
    size(a) == size(b) || throw(DimensionMismatch("audit masks must use the same grid"))
    a == b && return 0.0
    pa, pb = _boundary_lattice(a), _boundary_lattice(b)
    (!any(pa) || !any(pb)) && return Inf
    distances = Matrix{Float64}(undef, size(pa))
    _boundary_distance_field!(distances, pa)
    directed_ba = maximum(distances[p] for p in eachindex(pb) if pb[p])
    _boundary_distance_field!(distances, pb)
    directed_ab = maximum(distances[p] for p in eachindex(pa) if pa[p])
    return sqrt(max(directed_ab, directed_ba)) * PERIOD / size(pa, 1)
end

function _study_mask_metrics(mask, reference)
    union_count = count(mask .| reference)
    return (mask_jaccard=union_count == 0 ? 1.0 : count(mask .& reference) / union_count,
        boundary_displacement_deg=_study_boundary_distance(mask, reference),
        raster_topology=raster_topology(mask), reference_raster_topology=raster_topology(reference))
end

function _study_snapshot(analysis, observations, audit_grid_size)
    length(analysis.density_triangles) == length(analysis.alpha_triangles) ||
        error("alpha triangles must retain the density-triangle indexing")
    grid = periodic_density_grid(; grid_size=audit_grid_size)
    fine = periodic_density_grid(; grid_size=2audit_grid_size)
    density = _study_density_values(analysis, grid.phi_grid, grid.psi_grid)
    fine_density = _study_density_values(analysis, fine.phi_grid, fine.psi_grid)
    exact_mass = _exact_density_mass(analysis.density_triangles, analysis.point_density)
    results = NamedTuple[]
    for result in analysis.alpha_results
        mask = rasterize_alpha_mask(analysis.alpha_triangles, result.alpha,
            result.density_gate, grid.phi_grid, grid.psi_grid)
        fine_mask = rasterize_alpha_mask(analysis.alpha_triangles, result.alpha,
            result.density_gate, fine.phi_grid, fine.psi_grid)
        area = sum(tri.area for (tri, alpha_tri) in zip(analysis.density_triangles, analysis.alpha_triangles)
            if alpha_tri.grade <= result.alpha && alpha_tri.gate_density >= result.density_gate; init=0.0)
        coverage = _study_support_coverage(analysis, observations, result.alpha, result.density_gate)
        push!(results, (target=result.target, alpha=result.alpha, density_gate=result.density_gate,
            empirical_coverage=coverage, area_deg2=area, mask=mask, fine_mask=fine_mask,
            audit_area_deg2=count(mask) * grid.cell_area,
            fine_audit_area_deg2=count(fine_mask) * fine.cell_area,
            raster_topology=raster_topology(mask), fine_raster_topology=raster_topology(fine_mask)))
    end
    raw_mass = grid_density_mass(analysis.raw_del_density)
    smooth_mass = grid_density_mass(analysis.del_density)
    return (density=density, fine_density=fine_density, normalization_mass=exact_mass,
        phi_grid=grid.phi_grid, psi_grid=grid.psi_grid, cell_area=grid.cell_area,
        fine_cell_area=fine.cell_area, results=results,
        integrated_mass_relative_error=abs(raw_mass - exact_mass) / exact_mass,
        smoothing_mass_relative_error=abs(smooth_mass - raw_mass) / exact_mass,
        audit_density_mass_relative_error=abs(sum(density) * grid.cell_area - exact_mass) / exact_mass,
        fine_audit_density_mass_relative_error=abs(sum(fine_density) * fine.cell_area - exact_mass) / exact_mass)
end

function _compare_study_snapshots(candidate, reference, tolerances)
    l1 = sum(abs(candidate.density[i] / candidate.normalization_mass -
        reference.density[i] / reference.normalization_mass) for i in eachindex(candidate.density)) * candidate.cell_area
    fine_l1 = sum(abs(candidate.fine_density[i] / candidate.normalization_mass -
        reference.fine_density[i] / reference.normalization_mass) for i in eachindex(candidate.fine_density)) * candidate.fine_cell_area
    rows = NamedTuple[]
    for (a, b) in zip(candidate.results, reference.results)
        a.target == b.target || throw(ArgumentError("study targets differ"))
        mask_metrics = _study_mask_metrics(a.mask, b.mask)
        fine_metrics = _study_mask_metrics(a.fine_mask, b.fine_mask)
        boundary_audit_change = mask_metrics.boundary_displacement_deg == fine_metrics.boundary_displacement_deg ?
            0.0 : abs(mask_metrics.boundary_displacement_deg - fine_metrics.boundary_displacement_deg)
        audit_area_error = max(_relative_change(a.audit_area_deg2, a.area_deg2),
            _relative_change(b.audit_area_deg2, b.area_deg2),
            _relative_change(a.fine_audit_area_deg2, a.area_deg2),
            _relative_change(b.fine_audit_area_deg2, b.area_deg2))
        audit_stable = audit_area_error <= tolerances.audit_area_tolerance &&
            a.raster_topology == a.fine_raster_topology && b.raster_topology == b.fine_raster_topology &&
            boundary_audit_change <= tolerances.audit_boundary_tolerance_deg &&
            abs(l1 - fine_l1) <= tolerances.audit_density_l1_tolerance
        coverage_change = a.empirical_coverage - b.empirical_coverage
        area_change = _relative_change(a.area_deg2, b.area_deg2)
        gate_change = _relative_change(a.density_gate, b.density_gate)
        alpha_change = _relative_change(a.alpha, b.alpha)
        numeric_mass_valid = max(candidate.integrated_mass_relative_error,
            reference.integrated_mass_relative_error, candidate.smoothing_mass_relative_error,
            reference.smoothing_mass_relative_error) <= 1e-9
        agreement = fine_l1 <= tolerances.density_l1_tolerance &&
            abs(coverage_change) <= tolerances.coverage_tolerance && area_change <= tolerances.area_tolerance &&
            fine_metrics.boundary_displacement_deg <= tolerances.boundary_tolerance_deg &&
            a.fine_raster_topology == b.fine_raster_topology && gate_change <= tolerances.gate_relative_tolerance &&
            alpha_change <= tolerances.alpha_relative_tolerance && numeric_mass_valid
        push!(rows, (target=a.target, density_normalized_l1=l1,
            fine_audit_density_normalized_l1=fine_l1,
            integrated_mass_relative_error=candidate.integrated_mass_relative_error,
            smoothing_mass_relative_error=candidate.smoothing_mass_relative_error,
            audit_density_mass_relative_error=candidate.audit_density_mass_relative_error,
            fine_audit_density_mass_relative_error=candidate.fine_audit_density_mass_relative_error,
            density_gate=a.density_gate, reference_density_gate=b.density_gate,
            gate_relative_change=gate_change, alpha=a.alpha, reference_alpha=b.alpha,
            alpha_relative_change=alpha_change, empirical_coverage=a.empirical_coverage,
            reference_empirical_coverage=b.empirical_coverage, coverage_change=coverage_change,
            area_deg2=a.area_deg2, reference_area_deg2=b.area_deg2,
            relative_area_change=area_change, mask_metrics...,
            fine_audit_mask_jaccard=fine_metrics.mask_jaccard,
            fine_audit_boundary_displacement_deg=fine_metrics.boundary_displacement_deg,
            fine_raster_topology=a.fine_raster_topology, reference_fine_raster_topology=b.fine_raster_topology,
            audit_area_relative_error=audit_area_error, audit_boundary_change_deg=boundary_audit_change,
            audit_stable=audit_stable, agreement_within_tolerances=agreement,
            accepted=agreement && audit_stable))
    end
    return rows
end

"""
    density_convergence_study(geometry; points, grid_sizes=(96,192,384), ...)

Refine the conservative density grid and move its cell boundaries while holding
the certified triangulation, floor, smoothing width in degrees, and alpha rule
fixed. Density values and support masks are compared at common physical
coordinates. Raw observation containment and triangle-union area are geometric
queries, independent of the audit grid. Boundaries and topology remain raster
diagnostics, checked again at twice the audit resolution.

The finest unshifted grid is a comparison reference, not a proved exact answer.
`status=:converged_on_tested_grids` requires two consecutive accepted refinement
steps and accepted shifts at the two finest resolutions. Tolerances are explicit
engineering criteria, not statistical confidence bounds or coverage calibration.
Only the reference plot arrays are returned; every intermediate geometry is
shared and discarded from the compact report.
"""
function density_convergence_study(geometry; points, grid_sizes=(96, 192, 384),
        shift_fractions=(0.0, 0.25, 0.5), sigma_deg::Real=8.0,
        area_floor_quantile::Real=0.01, mass_targets=(0.98, 0.995), nsamples::Int=200,
        parsimony_fraction::Real=0.99, audit_grid_size::Int=768,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        density_l1_tolerance::Real=0.01, coverage_tolerance::Real=0.001,
        area_tolerance::Real=0.01, boundary_tolerance_deg::Real=2.0,
        gate_relative_tolerance::Real=0.01, alpha_relative_tolerance::Real=0.02,
        audit_area_tolerance::Real=0.005,
        audit_boundary_tolerance_deg::Real=PERIOD / audit_grid_size,
        audit_density_l1_tolerance::Real=0.001, progress=_quiet_progress)
    _check_study_geometry(geometry, points)
    _check_study_settings(; grid_sizes, shift_fractions, sigma_values=(sigma_deg,),
        area_floor_quantiles=(area_floor_quantile,), mass_targets, nsamples, audit_grid_size)
    tolerances = _study_tolerances(; density_l1_tolerance, coverage_tolerance,
        area_tolerance, boundary_tolerance_deg, gate_relative_tolerance,
        alpha_relative_tolerance, audit_area_tolerance,
        audit_boundary_tolerance_deg, audit_density_l1_tolerance)
    sizes = sort!(unique(Int.(collect(grid_sizes))))
    shifts = sort!(unique(Float64.(collect(shift_fractions))))
    targets = Tuple(Float64.(collect(mass_targets)))
    observations = _study_observations(geometry, points)
    function snapshot(n, sx, sy)
        origin = (DOMAIN_MIN + sx * PERIOD / n, DOMAIN_MIN + sy * PERIOD / n)
        progress("density convergence: $(n)x$(n), origin shift ($(sx), $(sy)) cells")
        artifact = delaunay_density_artifact(geometry; grid_size=n, grid_origin=origin,
            area_floor_quantile=area_floor_quantile, cache_dir=cache_dir,
            refresh_cache=refresh_cache, progress=_quiet_progress)
        analysis = alpha_analysis(artifact; sigma_deg=sigma_deg, mass_targets=targets,
            nsamples=nsamples, parsimony_fraction=parsimony_fraction)
        return _study_snapshot(analysis, observations, audit_grid_size)
    end
    reference = snapshot(last(sizes), 0.0, 0.0)
    zeros_by_size = Dict{Int,NamedTuple}(last(sizes) => reference)
    rows, origin_rows = NamedTuple[], NamedTuple[]
    for n in reverse(sizes)
        zero = get!(zeros_by_size, n) do
            snapshot(n, 0.0, 0.0)
        end
        for sx in shifts, sy in shifts
            candidate = sx == 0 && sy == 0 ? zero : snapshot(n, sx, sy)
            setting = (grid_size=n, origin_shift_phi_cells=sx, origin_shift_psi_cells=sy,
                origin_shift_phi_deg=sx * PERIOD / n, origin_shift_psi_deg=sy * PERIOD / n)
            append!(rows, [merge(setting, row) for row in _compare_study_snapshots(candidate, reference, tolerances)])
            append!(origin_rows, [merge(setting, row) for row in _compare_study_snapshots(candidate, zero, tolerances)])
        end
    end
    refinement_rows = NamedTuple[]
    for k in 1:(length(sizes) - 1)
        lower, upper = sizes[k], sizes[k + 1]
        append!(refinement_rows, [merge((coarse_grid_size=lower, fine_grid_size=upper,), row)
            for row in _compare_study_snapshots(zeros_by_size[lower], zeros_by_size[upper], tolerances)])
    end
    origin_summary = [(grid_size=n,
        max_density_normalized_l1=maximum(r.fine_audit_density_normalized_l1 for r in origin_rows if r.grid_size == n),
        max_absolute_coverage_change=maximum(abs(r.coverage_change) for r in origin_rows if r.grid_size == n),
        max_relative_area_change=maximum(r.relative_area_change for r in origin_rows if r.grid_size == n),
        max_boundary_displacement_deg=maximum(r.fine_audit_boundary_displacement_deg for r in origin_rows if r.grid_size == n),
        all_audits_stable=all(r.audit_stable for r in origin_rows if r.grid_size == n),
        all_shifts_accepted=all(r.accepted for r in origin_rows if r.grid_size == n)) for n in sizes]
    two_steps = length(sizes) >= 3
    enough_shifts = length(shifts) >= 2
    finest_two = sizes[max(1, length(sizes) - 1):end]
    last_two_coarse = sizes[max(1, length(sizes) - 2):max(1, length(sizes) - 1)]
    accepted_refinements = two_steps && all(r.accepted for r in refinement_rows if r.coarse_grid_size in last_two_coarse)
    accepted_origins = enough_shifts && all(r.all_shifts_accepted for r in origin_summary if r.grid_size in finest_two)
    status = accepted_refinements && accepted_origins ? :converged_on_tested_grids : :unresolved
    settings = (geometry_metadata=geometry.metadata, n_observations=size(points, 1),
        observation_hash=observations.point_hash, grid_sizes=sizes, shift_fractions=shifts,
        sigma_deg=Float64(sigma_deg), area_floor_quantile=Float64(area_floor_quantile),
        mass_targets=targets, nsamples=nsamples, parsimony_fraction=Float64(parsimony_fraction),
        audit_grid_size=audit_grid_size, fine_audit_grid_size=2audit_grid_size,
        tolerances=tolerances, reference_grid_size=last(sizes),
        grid_origin_convention=:lower_left_cell_boundary,
        density_comparison=:common_physical_coordinates,
        coverage_kind=observations.coverage_kind,
        coverage_evaluation=observations.evaluation,
        topology_kind=:closed_periodic_raster_cells)
    compact_reference = (phi_grid=reference.phi_grid, psi_grid=reference.psi_grid,
        density=reference.density,
        results=[(target=r.target, alpha=r.alpha, density_gate=r.density_gate, mask=r.mask) for r in reference.results])
    return (settings=settings, status=status, rows=rows, origin_rows=origin_rows,
        origin_summary=origin_summary, refinement_summary=refinement_rows,
        checks=(at_least_two_refinement_steps=two_steps, nonzero_shifts_tested=enough_shifts,
            last_two_refinements_accepted=accepted_refinements,
            finest_two_origin_studies_accepted=accepted_origins), reference=compact_reference)
end

function _stage_comparison(a, b)
    metrics = _study_mask_metrics(b.mask, a.mask)
    return (coverage_change=b.empirical_coverage - a.empirical_coverage,
        area_change_deg2=b.area_deg2 - a.area_deg2,
        relative_area_change=_relative_change(b.area_deg2, a.area_deg2), metrics...)
end

"""
    density_method_comparison(geometry; points, area_floor_quantiles=(0,.01), ...)

Measure four constructions separately: the unsmoothed linear density level
region, the smoothed reconstructed density level region, all triangles whose
vertices pass that level, and the selected alpha support. The final three use
one threshold and one smoothed field. The zero-width case evaluates the original
linear field, so it does not invent smoothing by reconstructing cell averages.

Rows distinguish exact linear/triangle geometry from audit-grid estimates.
Coverage is evaluated at the actual observations. Floor contrasts compare the
same smoothing scale, target, and stage against the explicitly included no-floor
case. Neither these contrasts nor the convergence study choose a new estimator
or claim calibrated out-of-sample coverage.
"""
function density_method_comparison(geometry; points, grid_size::Int=96,
        grid_origin=(-180.0, -180.0), area_floor_quantiles=(0.0, 0.01),
        sigma_values=(0.0, 4.0, 8.0, 12.0, 16.0), mass_targets=(0.98, 0.995),
        nsamples::Int=200, parsimony_fraction::Real=0.99, audit_grid_size::Int=768,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        progress=_quiet_progress)
    _check_study_geometry(geometry, points)
    _check_study_settings(; grid_sizes=(grid_size,), shift_fractions=(0.0,), sigma_values,
        area_floor_quantiles, mass_targets, nsamples, audit_grid_size)
    floors = sort!(unique(Float64.(collect(area_floor_quantiles))))
    sigmas = sort!(unique(Float64.(collect(sigma_values))))
    0.0 in floors || throw(ArgumentError("floor sensitivity must include the no-floor value 0"))
    0.0 in sigmas || throw(ArgumentError("smoothing sensitivity must include sigma=0"))
    targets = Tuple(Float64.(collect(mass_targets)))
    grid = periodic_density_grid(; grid_size=audit_grid_size)
    observations = _study_observations(geometry, points)
    rows, contrasts, floor_rows, floor_contrasts = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    no_floor_stages = Dict{Tuple{Float64,Float64,Symbol},NamedTuple}()
    selected_plot = nothing
    selected_sigma = 8.0 in sigmas ? 8.0 : last(sigmas)
    for floor_q in floors
        progress("density method comparison: area-floor quantile=$(floor_q)")
        artifact = delaunay_density_artifact(geometry; grid_size=grid_size, grid_origin=grid_origin,
            area_floor_quantile=floor_q, cache_dir=cache_dir,
            refresh_cache=refresh_cache, progress=_quiet_progress)
        push!(floor_rows, _study_floor_summary(artifact, floor_q))
        raw_values, covered = sample_triangle_density(artifact.density_triangles,
            artifact.point_density, grid.phi_grid, grid.psi_grid)
        all(covered) || error("certified geometry left an uncovered method audit point")
        raw_levels = [density_level_for_triangle_mass(artifact.density_triangles,
            artifact.point_density; mass_fraction=q) for q in targets]
        raw_stages = NamedTuple[]
        for (q, gate) in zip(targets, raw_levels)
            summary = triangle_superlevel_summary(artifact.density_triangles, artifact.point_density, gate)
            coverage = _study_raw_coverage(artifact, observations, gate)
            push!(raw_stages, (stage=:raw_density_level, target=q, density_gate=gate, alpha=nothing,
                empirical_coverage=coverage, area_deg2=summary.area, area_kind=:exact_linear_level_region,
                density_mass_fraction=summary.mass_fraction, mass_kind=:exact_linear_density,
                mask=BitMatrix(raw_values .>= gate)))
        end
        for sigma in sigmas
            progress("density method comparison: floor=$(floor_q), Gaussian sigma=$(sigma) degrees")
            analysis = alpha_analysis(artifact; sigma_deg=sigma, mass_targets=targets,
                nsamples=nsamples, parsimony_fraction=parsimony_fraction)
            length(analysis.density_triangles) == length(analysis.alpha_triangles) ||
                error("alpha triangles must retain the density-triangle indexing")
            density = sigma == 0 ? raw_values : _study_density_values(analysis, grid.phi_grid, grid.psi_grid)
            mass = _exact_density_mass(artifact.density_triangles, artifact.point_density)
            plot_targets = NamedTuple[]
            for (idx, result) in enumerate(analysis.alpha_results)
                gate, target = result.density_gate, result.target
                raw_stage = raw_stages[idx]
                level_mask = BitMatrix(density .>= gate)
                level_coverage = sigma == 0 ? raw_stage.empirical_coverage :
                    sum(observations.weights[i] for i in axes(observations.points, 1)
                        if _periodic_bilinear_value(analysis.del_density,
                            observations.points[i, 1], observations.points[i, 2]) >= gate; init=0) /
                        sum(observations.weights)
                level_stage = (stage=:smoothed_density_level, target=target, density_gate=gate, alpha=nothing,
                    empirical_coverage=level_coverage,
                    area_deg2=sigma == 0 ? raw_stage.area_deg2 : count(level_mask) * grid.cell_area,
                    area_kind=sigma == 0 ? :exact_linear_level_region : :audit_of_bilinear_level_region,
                    density_mass_fraction=sigma == 0 ? raw_stage.density_mass_fraction :
                        sum(density[level_mask]) * grid.cell_area / mass,
                    mass_kind=sigma == 0 ? :exact_linear_density : :audit_of_bilinear_density,
                    mask=level_mask)
                triangle_stages = NamedTuple[]
                for (stage, alpha) in ((:density_gated_triangles, Inf), (:alpha_support, result.alpha))
                    mask = rasterize_alpha_mask(analysis.alpha_triangles, alpha, gate, grid.phi_grid, grid.psi_grid)
                    area = sum(tri.area for (tri, alpha_tri) in zip(artifact.density_triangles, analysis.alpha_triangles)
                        if alpha_tri.grade <= alpha && alpha_tri.gate_density >= gate; init=0.0)
                    coverage = _study_support_coverage(analysis, observations, alpha, gate)
                    # The unsmoothed integral over any retained whole triangle is exact.
                    region_mass = sigma == 0 ? sum(tri.area * sum(artifact.point_density[i] for i in tri.base) / 3
                        for (tri, alpha_tri) in zip(artifact.density_triangles, analysis.alpha_triangles)
                        if alpha_tri.grade <= alpha && alpha_tri.gate_density >= gate; init=0.0) :
                        sum(density[mask]) * grid.cell_area
                    push!(triangle_stages, (stage=stage, target=target, density_gate=gate, alpha=alpha,
                        empirical_coverage=coverage, area_deg2=area, area_kind=:exact_triangle_union,
                        density_mass_fraction=region_mass / mass,
                        mass_kind=sigma == 0 ? :exact_linear_density : :audit_of_bilinear_density, mask=mask))
                end
                stages = (raw_stage, level_stage, triangle_stages...)
                for stage in stages
                    topology = raster_topology(stage.mask)
                    audit_area = count(stage.mask) * grid.cell_area
                    push!(rows, (area_floor_quantile=floor_q, sigma_deg=sigma,
                        target=target, stage=stage.stage, density_gate=stage.density_gate, alpha=stage.alpha,
                        empirical_coverage=stage.empirical_coverage, area_deg2=stage.area_deg2,
                        area_fraction=stage.area_deg2 / PERIOD^2, area_kind=stage.area_kind,
                        density_mass_fraction=stage.density_mass_fraction, mass_kind=stage.mass_kind,
                        audit_area_deg2=audit_area, audit_area_relative_error=stage.area_kind == :audit_of_bilinear_level_region ?
                            nothing : _relative_change(audit_area, stage.area_deg2),
                        raster_components=topology.components, raster_loops=topology.loops,
                        audit_resolution_validated=false))
                    key = (sigma, target, stage.stage)
                    if floor_q == 0
                        no_floor_stages[key] = stage
                    else
                        push!(floor_contrasts, (area_floor_quantile=floor_q, reference_area_floor_quantile=0.0,
                            sigma_deg=sigma, target=target, stage=stage.stage,
                            _stage_comparison(no_floor_stages[key], stage)...))
                    end
                end
                for (effect, from, to) in ((:smoothing, stages[1], stages[2]),
                        (:whole_triangle_gate, stages[2], stages[3]), (:alpha_radius, stages[3], stages[4]))
                    push!(contrasts, (area_floor_quantile=floor_q, sigma_deg=sigma, target=target,
                        effect=effect, from_stage=from.stage, to_stage=to.stage,
                        _stage_comparison(from, to)...))
                end
                push!(plot_targets, (target=target, masks=(raw_density_level=stages[1].mask,
                    smoothed_density_level=stages[2].mask,
                    density_gated_triangles=stages[3].mask, alpha_support=stages[4].mask)))
            end
            if floor_q == last(floors) && sigma == selected_sigma
                selected_plot = (area_floor_quantile=floor_q, sigma_deg=sigma,
                    phi_grid=grid.phi_grid, psi_grid=grid.psi_grid, targets=plot_targets)
            end
        end
    end
    settings = (geometry_metadata=geometry.metadata, n_observations=size(points, 1),
        observation_hash=observations.point_hash, grid_size=grid_size, grid_origin=Tuple(Float64.(grid_origin)),
        area_floor_quantiles=floors, sigma_values=sigmas, mass_targets=targets,
        nsamples=nsamples, parsimony_fraction=Float64(parsimony_fraction),
        audit_grid_size=audit_grid_size, audit_resolution_validated=false,
        coverage_kind=observations.coverage_kind,
        coverage_evaluation=observations.evaluation,
        topology_kind=:closed_periodic_raster_cells)
    return (; settings, rows, contrasts, floor_rows, floor_contrasts, selected_plot)
end
