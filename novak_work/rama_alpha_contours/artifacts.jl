# Geometry and conservative density artifacts. Geometry is cached independently
# of grid resolution, origin, flooring, and smoothing; density cache payloads
# contain no duplicate triangulation. All returned views share the input geometry.

function _matrix_sha1(mat::AbstractMatrix{<:Real})
    io = IOBuffer()
    write(io, Int64(size(mat, 1)), Int64(size(mat, 2)))
    write(io, reinterpret(UInt8, vec(Matrix{Float64}(mat))))
    return bytes2hex(sha1(take!(io)))
end

function _source_sha1()
    io = IOBuffer()
    for filename in ("RamaAlphaGeometry.jl", "RamaAlphaContours.jl",
            "rama_alpha_contours/integration.jl", "rama_alpha_contours/artifacts.jl",
            "rama_alpha_contours/studies.jl")
        write(io, filename, read(joinpath(@__DIR__, "..", filename)))
    end
    return bytes2hex(sha1(take!(io)))
end

function _geometry_source_sha1()
    io = IOBuffer()
    # These files own the periodic construction and point preprocessing. Changes
    # confined to integration or the experiment reports need not retriangulate.
    for filename in ("RamaAlphaGeometry.jl", "RamaAlphaContours.jl")
        write(io, filename, read(joinpath(@__DIR__, "..", filename)))
    end
    return bytes2hex(sha1(take!(io)))
end

function _load_cached_artifact(path, metadata, refresh_cache, progress)
    (path === nothing || refresh_cache || !isfile(path)) && return nothing
    cached = try
        deserialize(path)
    catch err
        progress("unreadable cache; rebuilding ($(typeof(err)))")
        nothing
    end
    if cached isa NamedTuple && haskey(cached, :metadata) && cached.metadata == metadata
        progress("cache hit: $(path)")
        return cached
    end
    return nothing
end

function _save_cached_artifact(path, artifact)
    path === nothing && return nothing
    mkpath(dirname(path))
    temporary, io = mktemp(dirname(path))
    try
        serialize(io, artifact)
        close(io)
        mv(temporary, path; force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    return nothing
end

"""
    delaunay_geometry_artifact(points; category, geometry_mode=:adaptive, ...)

Build or load periodic Delaunay geometry with original observation weights.
Grid size, grid origin, area flooring and smoothing do not enter this cache key.
Pass this result to `delaunay_density_artifact` repeatedly to reuse one geometry
through an entire convergence or estimator-parameter study.
"""
function delaunay_geometry_artifact(points::AbstractMatrix{<:Real};
        category::AbstractString, data_path::AbstractString="",
        geometry_mode::Symbol=:adaptive, ghost_margin::Real=45.0, seed::Integer=2026,
        use_grid_representatives::Bool=false, representative_grid::Int=256,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        progress=_quiet_progress)
    _check_points(points)
    geometry_mode in (:full, :adaptive, :fixed) || throw(ArgumentError("unknown geometry mode"))
    isfinite(ghost_margin) && 0 <= ghost_margin <= PERIOD || throw(ArgumentError("ghost_margin must lie in [0, 360]"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    representative_grid > 0 || throw(ArgumentError("representative_grid must be positive"))
    metadata = (cache_version="periodic-alpha-geometry-v4", source_sha1=_geometry_source_sha1(),
        julia_version=string(VERSION), triangulation_version=string(Base.pkgversion(RamaAlphaGeometry.DT)),
        data_path=isempty(data_path) ? "" : abspath(data_path), category=String(category),
        point_hash=_matrix_sha1(points), npoints=size(points, 1),
        geometry_mode, ghost_margin=geometry_mode == :full ? PERIOD : Float64(ghost_margin),
        seed=Int(seed), use_grid_representatives,
        representative_grid=use_grid_representatives ? representative_grid : nothing)
    path = cache_dir === nothing ? nothing : joinpath(cache_dir,
        "delaunay_geometry_$(bytes2hex(sha1(repr(metadata)))).jls")
    cached = _load_cached_artifact(path, metadata, refresh_cache, progress)
    if cached !== nothing
        return merge(cached, (cache_status=:loaded_from_cache, cache_file=path))
    end
    unique_pts, multiplicities, representative_summary = delaunay_input_points_with_counts(points;
        use_grid_representatives=use_grid_representatives,
        representative_grid=representative_grid, progress=progress)
    density_triangles, point_area, tiling_summary = periodic_delaunay_density_triangles(unique_pts;
        mode=geometry_mode, margin=ghost_margin, seed=seed, progress=progress)
    artifact = (; metadata, unique_pts, multiplicities, representative_summary,
        density_triangles, point_area, tiling_summary)
    _save_cached_artifact(path, artifact)
    return merge(artifact, (cache_status=path === nothing ? :computed_cache_disabled : :computed_and_saved,
        cache_file=path))
end

"""
    delaunay_density_artifact(geometry; grid_size=96, grid_origin=(-180,-180),
                             area_floor_quantile=0.01, ...)

Integrate the linear Delaunay density over periodic grid cells. `density_grid`
contains cell averages; `cell_mass` is their unnormalized mass. A zero floor
quantile disables flooring. No rescaling is used to hide integration error.

An uncertified fixed-margin construction may be inspected, but its density is
not usable for alpha analysis when cell coverage fails. Normal analyses and
convergence studies require certified geometry.
"""
function delaunay_density_artifact(geometry::NamedTuple;
        grid_size::Int=96, grid_origin=(-180.0,-180.0), area_floor_quantile::Real=0.01,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        progress=_quiet_progress)
    for key in (:metadata, :unique_pts, :multiplicities, :density_triangles, :point_area, :tiling_summary)
        hasproperty(geometry, key) || throw(ArgumentError("geometry artifact is missing $(key)"))
    end
    grid = periodic_density_grid(; grid_size=grid_size, grid_origin=grid_origin)
    isfinite(area_floor_quantile) && 0 <= area_floor_quantile <= 1 ||
        throw(ArgumentError("area_floor_quantile must lie in [0, 1]"))
    metadata = merge(geometry.metadata, (cache_version="conservative-alpha-density-v4",
        source_sha1=_source_sha1(), geometry_key=bytes2hex(sha1(repr(geometry.metadata))),
        grid=grid_size, grid_origin=grid.grid_origin,
        area_floor_quantile=Float64(area_floor_quantile), integration_method=:triangle_cell_exact,
        mass_tolerance=1.0e-9))
    path = cache_dir === nothing ? nothing : joinpath(cache_dir,
        "delaunay_density_$(bytes2hex(sha1(repr(metadata)))).jls")
    payload = _load_cached_artifact(path, metadata, refresh_cache, progress)
    cache_status = :loaded_from_cache
    if payload === nothing
        point_density, point_area_summary = delaunay_point_density(geometry.point_area, geometry.multiplicities;
            area_floor_quantile=area_floor_quantile, progress=progress)
        integrated = integrate_delaunay_density(geometry.density_triangles, point_density;
            grid_size=grid_size, grid_origin=grid.grid_origin, progress=progress)
        integrated.relative_mass_error <= metadata.mass_tolerance || error(
            "conservative density integration failed the total-mass check")
        if geometry.tiling_summary.mode != :fixed && !integrated.coverage_valid
            error("certified periodic geometry failed the per-cell area-coverage check")
        end
        integration_summary = (method=:triangle_cell_exact,
            exact_mass=integrated.exact_mass, deposited_mass=integrated.total_mass,
            relative_mass_error=integrated.relative_mass_error,
            max_relative_coverage_error=integrated.max_relative_coverage_error,
            coverage_valid=integrated.coverage_valid,
            geometric_area=sum(integrated.covered_area), expected_geometric_area=PERIOD^2,
            mass_tolerance=metadata.mass_tolerance)
        payload = (; metadata, point_density, point_area_summary,
            phi_grid=integrated.phi_grid, psi_grid=integrated.psi_grid,
            phi_edges=integrated.phi_edges, psi_edges=integrated.psi_edges,
            grid_origin=integrated.grid_origin,
            density_grid=integrated.density, cell_mass=integrated.cell_mass,
            covered_area=integrated.covered_area,
            density_covered=abs.(integrated.covered_area ./ integrated.cell_area .- 1) .<= 1.0e-8,
            integration_summary, normalization_mass=integrated.exact_mass)
        _save_cached_artifact(path, payload)
        cache_status = path === nothing ? :computed_cache_disabled : :computed_and_saved
    end
    return merge(geometry, payload, (geometry_metadata=geometry.metadata,
        geometry_cache_status=get(geometry, :cache_status, :provided_in_memory),
        geometry_cache_file=get(geometry, :cache_file, nothing), cache_status, cache_file=path))
end

"""Build geometry once, then integrate its density; pass a geometry artifact to reuse it explicitly."""
function delaunay_density_artifact(points::AbstractMatrix{<:Real};
        category::AbstractString, data_path::AbstractString="", grid_size::Int=96,
        grid_origin=(-180.0,-180.0), area_floor_quantile::Real=0.01,
        geometry_mode::Symbol=:adaptive, ghost_margin::Real=45.0, seed::Integer=2026,
        use_grid_representatives::Bool=false, representative_grid::Int=256,
        cache_dir::Union{Nothing,AbstractString}=nothing, refresh_cache::Bool=false,
        progress=_quiet_progress)
    periodic_density_grid(;grid_size=grid_size, grid_origin=grid_origin)
    isfinite(area_floor_quantile) && 0 <= area_floor_quantile <= 1 ||
        throw(ArgumentError("area_floor_quantile must lie in [0, 1]"))
    geometry = delaunay_geometry_artifact(points; category=category, data_path=data_path,
        geometry_mode=geometry_mode, ghost_margin=ghost_margin, seed=seed,
        use_grid_representatives=use_grid_representatives, representative_grid=representative_grid,
        cache_dir=cache_dir, refresh_cache=refresh_cache, progress=progress)
    return delaunay_density_artifact(geometry; grid_size=grid_size, grid_origin=grid_origin,
        area_floor_quantile=area_floor_quantile, cache_dir=cache_dir,
        refresh_cache=refresh_cache, progress=progress)
end

"""The unsmoothed conservative cell averages, with their original linear field retained."""
function raw_density_estimate(artifact)
    return (phi_grid=artifact.phi_grid, psi_grid=artifact.psi_grid,
        phi_edges=artifact.phi_edges, psi_edges=artifact.psi_edges,
        grid_origin=artifact.grid_origin,
        density=artifact.density_grid, cell_mass=artifact.cell_mass,
        covered=artifact.density_covered, covered_area=artifact.covered_area,
        point_density=artifact.point_density, point_area=artifact.point_area,
        point_area_summary=artifact.point_area_summary, triangles=artifact.density_triangles,
        exact_mass=artifact.normalization_mass, normalization_mass=artifact.normalization_mass,
        integration_summary=artifact.integration_summary, density_representation=:cell_average,
        reconstruction=:piecewise_linear_delaunay,
        method=:integrated_periodic_delaunay_area)
end

"""Apply smoothing, density gates, and the unchanged alpha-selection rule to one density artifact."""
function alpha_analysis(artifact; sigma_deg::Real=8.0, mass_targets=(0.98, 0.995),
        nsamples::Int=200, parsimony_fraction::Real=0.99)
    artifact.integration_summary.coverage_valid || error("periodic density does not cover every cell exactly once")
    raw = raw_density_estimate(artifact)
    est = smooth_delaunay_density_estimate(raw; sigma_deg=sigma_deg)
    levels = [density_level_for_mass(est; mass_fraction=q) for q in mass_targets]
    triangles = alpha_triangles_from_delaunay_density(artifact.density_triangles,
        artifact.point_density; gate_estimate=sigma_deg == 0 ? nothing : est)
    results = [choose_alpha_for_density_gate(est, triangles; target=q,
        density_gate=levels[i], nsamples=nsamples, parsimony_fraction=parsimony_fraction)
        for (i, q) in enumerate(mass_targets)]
    return (; artifact..., raw_del_density=raw, del_density=est,
        levels_del_mass=levels, alpha_triangles=triangles, alpha_results=results)
end
