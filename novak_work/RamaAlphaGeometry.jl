"""
Periodic Delaunay geometry for the alpha-contour notebook.

This module keeps the construction and its checks in one place so the notebook,
the small correctness fixtures, and the timing experiment use the same code.
Angles are in degrees on `[-180, 180)^2`.
"""
module RamaAlphaGeometry

using Random
import DelaunayTriangulation as DT

const _LOW = -180.0
const _HIGH = 180.0
const _PERIOD = 360.0
const _AREA = _PERIOD^2

"""One counterclockwise triangle in a periodic triangulation, with original point indices."""
const DensityTriangle = NamedTuple{(:a, :b, :c, :base, :area),
    Tuple{NTuple{2,Float64}, NTuple{2,Float64}, NTuple{2,Float64}, NTuple{3,Int}, Float64}}

function _big_circumcircle(a, b, c)
    return setprecision(BigFloat, 256) do
        aa, bb, cc = BigFloat.(a), BigFloat.(b), BigFloat.(c)
        bx, by = bb[1] - aa[1], bb[2] - aa[2]
        cx, cy = cc[1] - aa[1], cc[2] - aa[2]
        determinant = bx * cy - by * cx
        determinant == 0 && return (BigFloat(NaN), BigFloat(NaN), BigFloat(Inf), determinant)
        b2, c2 = bx^2 + by^2, cx^2 + cy^2
        ux = (cy * b2 - by * c2) / (2determinant)
        uy = (bx * c2 - cx * b2) / (2determinant)
        return aa[1] + ux, aa[2] + uy, hypot(ux, uy), determinant
    end
end

@inline function _circumcircle(a, b, c)
    # Work relative to a vertex; squaring absolute angular coordinates loses
    # accuracy for the very small triangles in densely sampled regions.
    bx, by = b[1] - a[1], b[2] - a[2]
    cx, cy = c[1] - a[1], c[2] - a[2]
    determinant = bx * cy - by * cx
    if abs(determinant) <= 32eps(Float64) * (abs(bx * cy) + abs(by * cx))
        return Float64.(_big_circumcircle(a, b, c))
    end
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / (2determinant)
    uy = (bx * c2 - cx * b2) / (2determinant)
    return a[1] + ux, a[2] + uy, hypot(ux, uy), determinant
end

"""Return the circumradius in degrees; collinear vertices have radius `Inf`."""
triangle_circumradius(a::NTuple{2,Float64}, b::NTuple{2,Float64}, c::NTuple{2,Float64}) =
    _circumcircle(a, b, c)[3]

function _validate_points(points)
    Base.require_one_based_indexing(points)
    size(points, 2) == 2 || throw(ArgumentError("points must have two columns (phi, psi)."))
    0 < size(points, 1) <= typemax(UInt32) ||
        throw(ArgumentError("provide at least one point and fewer than 2^32 points."))
    seen = Set{NTuple{2,Float64}}()
    sizehint!(seen, size(points, 1))
    for i in axes(points, 1)
        p = (Float64(points[i, 1]), Float64(points[i, 2]))
        p = map(x -> iszero(x) ? 0.0 : x, p)
        all(x -> isfinite(x) && _LOW <= x < _HIGH, p) ||
            throw(ArgumentError("wrap finite angles to [-180, 180) before building geometry."))
        p in seen && throw(ArgumentError("collapse duplicate coordinates and retain their multiplicities first."))
        push!(seen, p)
    end
    return nothing
end

function _tile_points(points, margin)
    coordinates = NTuple{2,Float64}[]
    base = Int[]
    shifts = NTuple{2,Int8}[]
    expected = min(9size(points, 1), ceil(Int, size(points, 1) * (1 + 2margin / _PERIOD)^2))
    sizehint!(coordinates, expected)
    sizehint!(base, expected)
    sizehint!(shifts, expected)
    lo, hi = _LOW - margin, _HIGH + margin
    for sy in -1:1, sx in -1:1, i in axes(points, 1)
        x, y = Float64(points[i, 1]) + sx * _PERIOD, Float64(points[i, 2]) + sy * _PERIOD
        lo <= x <= hi && lo <= y <= hi || continue
        push!(coordinates, (x, y))
        push!(base, i)
        push!(shifts, (Int8(sx), Int8(sy)))
    end
    return coordinates, base, shifts
end

function _is_two_dimensional(points)
    length(points) >= 3 || return false
    a, b = points[1], points[2]
    for c in @view(points[3:end])
        (_, _, _, determinant) = _circumcircle(a, b, c)
        determinant != 0.0 && return true
    end
    return false
end

@inline function _central_circumcenter(x, y, a, b, c)
    # Exact seams are common in grid fixtures. Re-evaluate ownership there at
    # higher precision rather than shifting a half-open boundary by a tolerance.
    if min(abs(x - _LOW), abs(x - _HIGH), abs(y - _LOW), abs(y - _HIGH)) < 1.0e-8
        ux, uy, _, _ = _big_circumcircle(a, b, c)
        return _LOW <= ux < _HIGH && _LOW <= uy < _HIGH
    end
    return _LOW <= x < _HIGH && _LOW <= y < _HIGH
end

@inline function _edge_code(i, j, si, sj)
    dx, dy = Int(sj[1]) - Int(si[1]), Int(sj[2]) - Int(si[2])
    forward = i < j || (i == j && (dx > 0 || (dx == 0 && dy > 0)))
    if !forward
        i, j, dx, dy = j, i, -dx, -dy
    end
    # A periodic edge needs its translation as well as its endpoint indices:
    # there can be several edges between the same two vertices, and even loops.
    # Packing these small integer offsets avoids millions of dictionary entries.
    return (UInt128(i) << 39) | (UInt128(j) << 7) |
        (UInt128(dx + 2) << 4) | (UInt128(dy + 2) << 1) | UInt128(forward)
end

function _unpaired_edges!(codes)
    sort!(codes)
    unpaired = 0
    i = 1
    while i <= length(codes)
        key = codes[i] >> 1
        j = i + 1
        while j <= length(codes) && codes[j] >> 1 == key
            j += 1
        end
        if j - i != 2 || (codes[i] & 1) == (codes[i + 1] & 1)
            unpaired += 1
        end
        i = j
    end
    return unpaired
end

function _geometry_attempt(points, margin, seed, progress)
    start = time_ns()
    coords, base, shifts = _tile_points(points, margin)
    progress("ghost margin=$(margin) degrees: triangulating $(length(coords)) periodic points")
    n = size(points, 1)
    rows = DensityTriangle[]
    area_acc = zeros(n)
    angle_acc = zeros(n)
    edge_codes = UInt128[]
    sizehint!(rows, 2n)
    sizehint!(edge_codes, 6n)
    failed_circumdisks = 0
    invalid_triangles = 0
    min_circle_clearance = Inf
    total_area = 0.0

    if _is_two_dimensional(coords)
        tri = DT.triangulate(coords; rng=Random.Xoshiro(seed))
        for t in DT.each_solid_triangle(tri)
            ia, ib, ic = t
            a, b, c = coords[ia], coords[ib], coords[ic]
            x, y, radius, determinant = _circumcircle(a, b, c)
            if !isfinite(radius)
                invalid_triangles += 1
                continue
            end
            _central_circumcenter(x, y, a, b, c) || continue
            if determinant < 0.0
                ib, ic = ic, ib
                b, c = c, b
                determinant = -determinant
            end
            area = determinant / 2
            ba, bb, bc = base[ia], base[ib], base[ic]
            push!(rows, (a=a, b=b, c=c, base=(ba, bb, bc), area=area))
            total_area += area
            # Repeated base indices are intentional on a sparsely sampled torus.
            # Each corner contributes separately to the same vertex's star.
            area_acc[ba] += area / 3
            area_acc[bb] += area / 3
            area_acc[bc] += area / 3
            angle_acc[ba] += atan(determinant, (b[1]-a[1])*(c[1]-a[1]) + (b[2]-a[2])*(c[2]-a[2]))
            angle_acc[bb] += atan(determinant, (a[1]-b[1])*(c[1]-b[1]) + (a[2]-b[2])*(c[2]-b[2]))
            angle_acc[bc] += atan(determinant, (a[1]-c[1])*(b[1]-c[1]) + (a[2]-c[2])*(b[2]-c[2]))
            push!(edge_codes, _edge_code(ba, bb, shifts[ia], shifts[ib]))
            push!(edge_codes, _edge_code(bb, bc, shifts[ib], shifts[ic]))
            push!(edge_codes, _edge_code(bc, ba, shifts[ic], shifts[ia]))

            clearance = min(x - (_LOW-margin), (_HIGH+margin) - x,
                            y - (_LOW-margin), (_HIGH+margin) - y) - radius
            min_circle_clearance = min(min_circle_clearance, clearance)
            # Strict containment leaves room for floating-point arithmetic;
            # a marginal case escalates instead of receiving a false certificate.
            clearance > 256eps(Float64) * max(_PERIOD, radius) || (failed_circumdisks += 1)
        end
    end

    unpaired_edges = _unpaired_edges!(edge_codes)
    missing_vertices = count(iszero, area_acc)
    max_star_angle_error = maximum(abs.(angle_acc .- 2pi))
    area_relative_error = abs(total_area - _AREA) / _AREA
    certified = failed_circumdisks == 0 && invalid_triangles == 0 &&
        unpaired_edges == 0 && missing_vertices == 0 && length(rows) == 2n &&
        max_star_angle_error <= 1.0e-7 && area_relative_error <= 1.0e-9
    summary = (;
        margin, n_tiled_points=length(coords), n_triangles=length(rows),
        certified, failed_circumdisks, invalid_triangles, unpaired_edges,
        missing_vertices, max_star_angle_error, total_area, area_relative_error,
        min_circle_clearance, elapsed_seconds=(time_ns() - start) / 1.0e9,
    )
    progress("geometry check: certified=$(certified), area=$(total_area), unsafe circles=$(failed_circumdisks), unpaired edges=$(unpaired_edges)")
    return rows, area_acc, summary
end

"""
    periodic_delaunay_density_triangles(points; mode=:adaptive, margin=45.0,
                                        seed=20260922, progress=(_ -> nothing))

Build one representative of every triangle on the 360-degree square torus.
`points` is an `N × 2` matrix of finite, distinct coordinates in `[-180,180)`;
collapse duplicates beforehand and keep their multiplicities separately.

Return `(triangles, point_areas, summary)`. A point area is one third of the
areas of all triangles incident to that point, counting each periodic corner.
Thus unregularized densities `multiplicity ./ point_areas` integrate exactly to
the total multiplicity under linear interpolation on the triangles.

* `:adaptive` starts with `margin` degrees of neighboring points and doubles the
  margin until all checks pass, falling back to the full nine tiles at 360.
* `:full` uses the central tile and all eight neighbors as the reference.
* `:fixed` performs one requested margin only and reports failed checks without
  throwing. This mode is an explicitly approximate comparison, not a fallback.

We choose triangles by their circumcenters in the central half-open square.
A triangle is safe from omitted periodic points when its entire circumdisk is
inside the fully populated rectangle. This is checked for the whole density
triangulation, before any alpha or density cutoff. Edge pairing, each vertex's
full angle of `2π`, the `2N` face count, and the total area `360²` also check that
the selected triangles form one complete torus. `:full` throws if these checks
fail. All checks are numerical; tolerances are recorded in the source.

The fixed `seed` controls Delaunay insertion order. Exactly cocircular points
allow more than one valid diagonal; a certificate guarantees periodic geometry,
not identical diagonals across independently built tied triangulations.
"""
function periodic_delaunay_density_triangles(points::AbstractMatrix{<:Real};
        mode::Symbol=:adaptive, margin::Real=45.0, seed::Integer=20260922,
        label=nothing, progress=(_ -> nothing))
    mode in (:adaptive, :full, :fixed) ||
        throw(ArgumentError("mode must be :adaptive, :full, or :fixed."))
    margin_f = Float64(margin)
    isfinite(margin_f) && 0 <= margin_f <= _PERIOD ||
        throw(ArgumentError("margin must lie in [0, 360] degrees."))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative."))
    _validate_points(points)
    attempted_margin = mode == :full ? _PERIOD : margin_f
    attempts = NamedTuple[]
    while true
        rows, area_acc, attempt = _geometry_attempt(points, attempted_margin, seed, progress)
        push!(attempts, attempt)
        if attempt.certified || mode == :fixed
            summary = merge(attempt, (; mode, requested_margin=margin_f, seed,
                label, attempts, used_full_tiling=attempted_margin == _PERIOD,
                triangle_representative=:circumcenter))
            return rows, area_acc, summary
        end
        attempted_margin == _PERIOD && error(
            "Full periodic triangulation did not pass its geometry checks: $(attempt). " *
            "Do not use this geometry for density or contour inference.")
        attempted_margin = min(_PERIOD, max(1.0, 2attempted_margin))
    end
end

"""
    foreach_periodic_triangle(f, triangle; xlimits=(-180,180), ylimits=(-180,180))

Call `f(a, b, c)` for every translation by multiples of 360 whose bounding box
meets the requested rectangle. Use the same callback for density interpolation
and alpha-mask filling so pieces across either plotting seam are retained.
Triangle interiors, rather than bounding boxes, must be tested by the caller.
"""
function foreach_periodic_triangle(f, tri; xlimits=(_LOW, _HIGH), ylimits=(_LOW, _HIGH))
    xlo, xhi = Float64.(xlimits)
    ylo, yhi = Float64.(ylimits)
    all(isfinite, (xlo, xhi, ylo, yhi)) && xlo <= xhi && ylo <= yhi ||
        throw(ArgumentError("limits must be finite ordered pairs."))
    a, b, c = tri.a, tri.b, tri.c
    minx, maxx = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
    miny, maxy = min(a[2], b[2], c[2]), max(a[2], b[2], c[2])
    ix = ceil(Int, (xlo-maxx) / _PERIOD):floor(Int, (xhi-minx) / _PERIOD)
    iy = ceil(Int, (ylo-maxy) / _PERIOD):floor(Int, (yhi-miny) / _PERIOD)
    for j in iy, i in ix
        dx, dy = i * _PERIOD, j * _PERIOD
        f((a[1]+dx, a[2]+dy), (b[1]+dx, b[2]+dy), (c[1]+dx, c[2]+dy))
    end
    return nothing
end

end # module
