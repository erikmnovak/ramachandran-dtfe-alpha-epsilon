module RamaAlphaUtils

using JSON3
using Statistics

export DensityEstimate,
       load_phi_psi_jsonl,
       load_phi_psi_by_category_jsonl,
       torus_grid,
       torus_delta,
       torus_distance,
       point_counts_torus,
       gaussian_kde_torus,
       binned_gaussian_kde_torus,
       density_estimate,
       density_mass,
       density_level_for_mass

const DOMAIN = (-180.0, 180.0)
const PERIOD = DOMAIN[2] - DOMAIN[1]
const DEFAULT_BINNED_WORK_THRESHOLD = 20_000_000

"""
    DensityEstimate

Scalar density image on the Ramachandran torus.

The matrix is indexed as `density[psi_index, phi_index]`; both axes are
periodic and use degree units.
"""
struct DensityEstimate
    phi_grid::Vector{Float64}
    psi_grid::Vector{Float64}
    density::Matrix{Float64}
    method::Symbol
    parameters::NamedTuple
end

@inline function _json_get(obj, key::Symbol)
    return hasproperty(obj, key) ? getproperty(obj, key) : nothing
end

@inline function _is_missing_angle(x)
    return x === nothing || x isa Missing
end

"""
    load_phi_psi_jsonl(path; category_field=nothing, category=nothing, limit=nothing)

Load `(phi, psi)` angle pairs from one of the local Prisant JSONL files.

Rows with null `phi` or `psi` are dropped. If `category_field` and `category`
are supplied, only matching rows are kept.
"""
function load_phi_psi_jsonl(path;
                            category_field::Union{Nothing,Symbol}=nothing,
                            category=nothing,
                            limit::Union{Nothing,Integer}=nothing)
    phis = Float64[]
    psis = Float64[]
    nmax = limit === nothing ? typemax(Int) : Int(limit)
    nmax >= 0 || throw(ArgumentError("limit must be nonnegative."))
    open(string(path), "r") do io
        for line in eachline(io)
            obj = JSON3.read(line)
            if category_field !== nothing
                raw = _json_get(obj, category_field)
                raw === nothing && continue
                category === nothing || string(raw) == string(category) || continue
            end
            phi = _json_get(obj, :phi)
            psi = _json_get(obj, :psi)
            (_is_missing_angle(phi) || _is_missing_angle(psi)) && continue
            push!(phis, Float64(phi))
            push!(psis, Float64(psi))
            length(phis) >= nmax && break
        end
    end
    pts = Matrix{Float64}(undef, length(phis), 2)
    @inbounds for i in eachindex(phis)
        pts[i, 1] = phis[i]
        pts[i, 2] = psis[i]
    end
    return pts
end

"""
    load_phi_psi_by_category_jsonl(path, categories; category_field=:rama_category, limit=nothing)

Load `(phi, psi)` angle pairs from a local Prisant JSONL file in one pass,
split by category. Rows with null `phi` or `psi` are dropped. The optional
`limit` is a per-category limit, which is useful for notebook smoke checks.
"""
function load_phi_psi_by_category_jsonl(path,
                                        categories;
                                        category_field::Symbol=:rama_category,
                                        limit::Union{Nothing,Integer}=nothing)
    labels = String[string(c) for c in categories]
    wanted = Set(labels)
    phis = Dict(label => Float64[] for label in labels)
    psis = Dict(label => Float64[] for label in labels)
    nmax = limit === nothing ? typemax(Int) : Int(limit)
    nmax >= 0 || throw(ArgumentError("limit must be nonnegative."))
    open(string(path), "r") do io
        for line in eachline(io)
            obj = JSON3.read(line)
            raw = _json_get(obj, category_field)
            raw === nothing && continue
            label = string(raw)
            label in wanted || continue
            length(phis[label]) >= nmax && continue
            phi = _json_get(obj, :phi)
            psi = _json_get(obj, :psi)
            (_is_missing_angle(phi) || _is_missing_angle(psi)) && continue
            push!(phis[label], Float64(phi))
            push!(psis[label], Float64(psi))
        end
    end
    out = Dict{String,Matrix{Float64}}()
    for label in labels
        mat = Matrix{Float64}(undef, length(phis[label]), 2)
        @inbounds for i in eachindex(phis[label])
            mat[i, 1] = phis[label][i]
            mat[i, 2] = psis[label][i]
        end
        out[label] = mat
    end
    return out
end

function torus_grid(grid_size::Integer)
    n = Int(grid_size)
    n > 0 || throw(ArgumentError("grid_size must be positive."))
    step = PERIOD / n
    return collect(range(DOMAIN[1] + step / 2; step=step, length=n))
end

@inline function torus_delta(a::Real, b::Real)
    d = abs(Float64(a) - Float64(b))
    return min(d, PERIOD - d)
end

@inline function torus_distance(p1::Real, q1::Real, p2::Real, q2::Real)
    return hypot(torus_delta(p1, p2), torus_delta(q1, q2))
end

function _check_points(points::AbstractMatrix{<:Real})
    size(points, 2) == 2 ||
        throw(ArgumentError("Ramachandran points must be an n x 2 matrix of (phi, psi)."))
    size(points, 1) > 0 ||
        throw(ArgumentError("Ramachandran point set is empty."))
    return nothing
end

@inline function _grid_bin_index(angle::Real, n::Int)
    x = mod(Float64(angle) - DOMAIN[1], PERIOD)
    idx = Int(floor(x / PERIOD * n)) + 1
    return idx > n ? n : idx
end

"""
    point_counts_torus(points; grid_size=64)

Bin Ramachandran angle pairs onto a periodic `T^2` grid. The result is indexed
as `counts[psi_index, phi_index]`.
"""
function point_counts_torus(points::AbstractMatrix{<:Real}; grid_size::Integer=64)
    _check_points(points)
    n = Int(grid_size)
    n > 0 || throw(ArgumentError("grid_size must be positive."))
    counts = zeros(Float64, n, n)
    @inbounds for p in axes(points, 1)
        i = _grid_bin_index(points[p, 1], n)
        j = _grid_bin_index(points[p, 2], n)
        counts[j, i] += 1.0
    end
    return counts
end

"""
    gaussian_kde_torus(points; grid_size=64, bw_deg=8.0)

Gaussian KDE on `T^2`, using the minimum-image torus distance in each angular
coordinate. This helper is used only for optional reference overlays in the
alpha-shape notebook.
"""
function gaussian_kde_torus(points::AbstractMatrix{<:Real};
                            grid_size::Integer=64,
                            bw_deg::Real=8.0)
    _check_points(points)
    bw = Float64(bw_deg)
    bw > 0 || throw(ArgumentError("bw_deg must be positive."))
    phi_grid = torus_grid(grid_size)
    psi_grid = torus_grid(grid_size)
    n = size(points, 1)
    density = zeros(Float64, length(psi_grid), length(phi_grid))
    inv2var = 1.0 / (2.0 * bw * bw)
    norm = 1.0 / (2.0 * pi * bw * bw * n)
    @inbounds for p in 1:n
        phi0 = Float64(points[p, 1])
        psi0 = Float64(points[p, 2])
        for j in eachindex(psi_grid)
            dy = torus_delta(psi_grid[j], psi0)
            ky = exp(-(dy * dy) * inv2var)
            for i in eachindex(phi_grid)
                dx = torus_delta(phi_grid[i], phi0)
                density[j, i] += ky * exp(-(dx * dx) * inv2var)
            end
        end
    end
    density .*= norm
    return DensityEstimate(phi_grid, psi_grid, density, :gaussian,
                           (; grid_size=Int(grid_size), bw_deg=bw, backend=:exact))
end

"""
    binned_gaussian_kde_torus(points; grid_size=64, bw_deg=8.0, cutoff_sigma=4.0)

Fast binned Gaussian KDE on `T^2`. Points are first accumulated into periodic
grid bins, then convolved with the separable Gaussian kernel on the torus.
"""
function binned_gaussian_kde_torus(points::AbstractMatrix{<:Real};
                                   grid_size::Integer=64,
                                   bw_deg::Real=8.0,
                                   cutoff_sigma::Real=4.0)
    _check_points(points)
    n = Int(grid_size)
    bw = Float64(bw_deg)
    cutoff = Float64(cutoff_sigma)
    n > 0 || throw(ArgumentError("grid_size must be positive."))
    bw > 0 || throw(ArgumentError("bw_deg must be positive."))
    cutoff > 0 || throw(ArgumentError("cutoff_sigma must be positive."))
    counts = point_counts_torus(points; grid_size=n)
    step = PERIOD / n
    radius = min((n - 1) ÷ 2, ceil(Int, cutoff * bw / step))
    offsets = collect(-radius:radius)
    inv2var = 1.0 / (2.0 * bw * bw)
    weights = [exp(-((o * step)^2) * inv2var) for o in offsets]
    tmp = zeros(Float64, n, n)
    density = zeros(Float64, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            acc = 0.0
            for k in eachindex(offsets)
                acc += counts[j, mod1(i - offsets[k], n)] * weights[k]
            end
            tmp[j, i] = acc
        end
    end
    @inbounds for j in 1:n
        for i in 1:n
            acc = 0.0
            for k in eachindex(offsets)
                acc += tmp[mod1(j - offsets[k], n), i] * weights[k]
            end
            density[j, i] = acc
        end
    end
    density .*= 1.0 / (2.0 * pi * bw * bw * size(points, 1))
    phi_grid = torus_grid(n)
    psi_grid = torus_grid(n)
    return DensityEstimate(phi_grid, psi_grid, density, :gaussian,
                           (; grid_size=n, bw_deg=bw,
                              backend=:binned, cutoff_sigma=cutoff))
end

function density_estimate(points::AbstractMatrix{<:Real};
                          kde::Symbol=:gaussian,
                          backend::Symbol=:auto,
                          grid_size::Integer=64,
                          bw_deg::Real=8.0,
                          binned_work_threshold::Integer=DEFAULT_BINNED_WORK_THRESHOLD)
    kde === :gaussian ||
        throw(ArgumentError("RamaAlphaUtils only provides :gaussian KDE for reference overlays."))
    backend in (:auto, :exact, :binned) ||
        throw(ArgumentError("backend must be :auto, :exact, or :binned."))
    work = size(points, 1) * Int(grid_size) * Int(grid_size)
    backend_eff = backend === :auto && work > Int(binned_work_threshold) ? :binned : backend
    return backend_eff === :binned ?
           binned_gaussian_kde_torus(points; grid_size=grid_size, bw_deg=bw_deg) :
           gaussian_kde_torus(points; grid_size=grid_size, bw_deg=bw_deg)
end

function density_mass(est::DensityEstimate)
    dx = PERIOD / length(est.phi_grid)
    dy = PERIOD / length(est.psi_grid)
    return sum(est.density) * dx * dy
end

function density_level_for_mass(est::DensityEstimate; mass_fraction::Real=0.98)
    f = Float64(mass_fraction)
    0 < f <= 1 || throw(ArgumentError("mass_fraction must lie in (0, 1]."))
    vals = collect(vec(est.density))
    order = sortperm(vals; rev=true)
    dx = PERIOD / length(est.phi_grid)
    dy = PERIOD / length(est.psi_grid)
    target = f * sum(vals) * dx * dy
    accum = 0.0
    for idx in order
        accum += vals[idx] * dx * dy
        accum >= target && return vals[idx]
    end
    return minimum(vals)
end

end
