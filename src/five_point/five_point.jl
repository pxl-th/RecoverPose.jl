function pre_divide(points1, points2, K1, K2)
    p1 = Vector{SVector{2, Float64}}(undef, length(points1))
    p2 = Vector{SVector{2, Float64}}(undef, length(points1))

    @inbounds fx1, fy1, px1, py1 = K1[1, 1], K1[2, 2], K1[1, 3], K1[2, 3]
    @inbounds fx2, fy2, px2, py2 = K2[1, 1], K2[2, 2], K2[1, 3], K2[2, 3]
    @inbounds @simd for i in 1:length(points1)
        p1[i] = SVector{2, Float64}(
            (points1[i][1] - px1) / fx1,
            (points1[i][2] - py1) / fy1)
        p2[i] = SVector{2, Float64}(
            (points2[i][1] - px2) / fx2,
            (points2[i][2] - py2) / fy2)
    end
    p1, p2
end

"""
```julia
five_point(points1, points2, K1, K2; max_repr_error = 1.0)
```

Compute Essential matrix and recover pose from it for a given set of points.
This function accepts five points, if you have more of them,
consider using [`five_point_ransac`](@ref) version.

# Arguments:

- `points1`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the first image.
- `points2`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the second image.
- `K1`: Intrinsic matrix for the first set of points.
- `K2`: Intrinsic matrix for the second set of points.
- `max_repr_error`: Maximum allowed reprojection error for a point to be
    considered as inlier. Default is `1.0`.

# Returns:

`n_inliers, (E, P, inliers, repr_error)`:

- number of inliers
- tuple: essential matrix, pose matrix, boolean vector of inliers, error value.

!!! note

    Pose matrix transforms points from the first camera to the second camera.
"""
function five_point(points1, points2, K1, K2; max_repr_error = 1.0)
    p1, p2 = pre_divide(points1, points2, K1, K2)
    candidates = five_point_candidates(p1, p2)
    select_candidates(candidates, points1, points2, K1, K2; max_repr_error)
end

"""
```julia
five_point_ransac(
    points1, points2, K1, K2; max_repr_error = 1.0, ransac_kwargs...)
```

Compute Essential matrix and recover pose from it using RANSAC scheme.

# Arguments:

- `points1`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the first image.
- `points2`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the second image.
- `K1`: Intrinsic matrix for the first set of points.
- `K2`: Intrinsic matrix for the second set of points.
- `max_repr_error`: Maximum allowed reprojection error for a point to be
    considered as inlier. Default is `1.0`.
- `ransac_kwargs...`: Keyword arguments passed to [`ransac`](@ref).

# Returns:

`n_inliers, (E, P, inliers, repr_error)`:

- number of inliers
- tuple: essential matrix, pose matrix, boolean vector of inliers, error value.

!!! note

    Pose matrix transforms points from the first camera to the second camera.
"""
@inline five_point_ransac(pixels1, pixels2, K1, K2; max_repr_error = 1.0, ransac_kwargs...) =
    five_point_ransac(pixels1, pixels2, K1, K2, GEEV4x4Cache(); max_repr_error, ransac_kwargs...)

function five_point_ransac(
    pixels1, pixels2, K1, K2, cache::GEEV4x4Cache;
    max_repr_error = 1.0, ransac_kwargs...,
)
    pd1, pd2 = pre_divide(pixels1, pixels2, K1, K2)
    five_point_ransac(
        pixels1, pixels2, pd1, pd2, K1, K2, cache;
        max_repr_error, ransac_kwargs...)
end

function five_point_ransac(
    pixels1, pixels2, pd1, pd2, K1, K2, cache::GEEV4x4Cache;
    max_repr_error = 1.0, ransac_kwargs...,
)
    sample_selection(sample_ids) = (pd1[sample_ids], pd2[sample_ids])
    rank(models; sample_ids) = select_candidates(
        models, pixels1, pixels2, K1, K2, cache; max_repr_error, sample_ids)
    ransac(
        sample_selection, five_point_candidates, rank,
        length(pd1), 5; ransac_kwargs...)
end

"""
```julia
essential_ransac(pixels1, pixels2, K1, K2; threshold = 1.0, ransac_kwargs...)
```

Compute Essential matrix using the RANASC scheme.

# Arguments:

- `pixels1`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the first image.
- `pixels2`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the second image.
- `K1`: Intrinsic matrix for the first set of points.
- `K2`: Intrinsic matrix for the second set of points.
- `threshold`: Maximum error for the epipolar constraint. Default is `1.0`.
- `ransac_kwargs...`: Keyword arguments passed to [`ransac`](@ref).

# Returns:

`n_inliers, (E, P, inliers, repr_error)`:

- number of inliers
- tuple: essential matrix, boolean vector of inliers, error value.
"""
function essential_ransac(
    pixels1, pixels2, K1, K2; threshold = 1.0, ransac_kwargs...,
)
    pd1, pd2 = pre_divide(pixels1, pixels2, K1, K2)
    essential_ransac(
        pixels1, pixels2, pd1, pd2, K1, K2; threshold, ransac_kwargs...)
end

function essential_ransac(
    pixels1, pixels2, pd1, pd2, K1, K2; threshold = 1.0, ransac_kwargs...,
)
    threshold /= (K1[1, 1] + K1[2, 2]) / 2.0
    sample_selection(sample_ids) = (pd1[sample_ids], pd2[sample_ids])
    rank(models; sample_ids) = select_candidate(
        models, pd1, pd2; threshold, sample_ids)
    ransac(
        sample_selection, five_point_candidates, rank,
        length(pd1), 5; ransac_kwargs...)
end

struct FivePointCache
    F::Matrix{Float64}
    M::Matrix{Float64}
    Z::Matrix{Float64}
end

function FivePointCache(n_points)
    F = Matrix{Float64}(undef, n_points, 9)
    M = Matrix{Float64}(undef, 10, 20)
    Z = Matrix{Float64}(undef, 10, 10)
    FivePointCache(F, M, Z)
end

"""
Compute essential matrix using Five-Point algorithm.

# Arguments:

- `p1`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.
- `p2`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.

# Returns:

Vector of `3x3` essential matrix candidates.
"""
@inline five_point_candidates(p1, p2) =
    five_point_candidates!(FivePointCache(length(p1)), p1, p2)

function five_point_candidates!(cache::FivePointCache, p1, p2)
    @polyvar z

    V = null_space!(cache.F, p1, p2)
    E1 = SMatrix{3, 3, Float64, 9}(V[:, 6,]...)'
    E2 = SMatrix{3, 3, Float64, 9}(V[:, 7]...)'
    E3 = SMatrix{3, 3, Float64, 9}(V[:, 8]...)'
    E4 = SMatrix{3, 3, Float64, 9}(V[:, 9]...)'
    rref_M = compute_rref!(cache.M, E1, E2, E3, E4)

    B11, B12, B13 = to_polynoms(@view(rref_M[5, 11:20]), @view(rref_M[6, 11:20]), z)
    B21, B22, B23 = to_polynoms(@view(rref_M[7, 11:20]), @view(rref_M[8, 11:20]), z)
    B31, B32, B33 = to_polynoms(@view(rref_M[9, 11:20]), @view(rref_M[10, 11:20]), z)

    # Calculate determinant.
    P1 = B23 * B12 - B13 * B22
    P2 = B13 * B21 - B23 * B11
    P3 = B11 * B22 - B12 * B21
    det_B = P1 * B31 + P2 * B32 + P3 * B33

    # Normalize the coefficient of the highest order.
    coefficient(det_B, z^10) ≉ 0 && (det_B /= coefficient(det_B, z^10);)

    # Extract real roots of polynomial with companion matrix.
    fill!(cache.Z, 0.0)
    sub_Z = @view(cache.Z[1:9, 2:10])
    sub_Z[diagind(sub_Z)] .= 1.0
    @inbounds cache.Z[10, :] .= -coefficients(det_B)[end:-1:2]
    sol_z = Float64[real(s) for s in eigvals!(cache.Z) if isreal(s)]
    isempty(sol_z) && return SMatrix{3, 3, Float64, 9}[]

    # Compute x & y.
    z6 = Matrix{Float64}(undef, length(sol_z), 7)
    z7 = Matrix{Float64}(undef, length(sol_z), 8)
    @inbounds z6[:, 7] .= 1.0
    @inbounds z7[:, 8] .= 1.0
    @inbounds z7[:, 1] .= sol_z.^7
    @inbounds @simd for i in 1:6
        v = sol_z .^ i
        z6[:, 7 - i] .= v
        z7[:, 8 - i] .= v
    end

    P1z = z7 * coefficients(P1)
    P2z = z7 * coefficients(P2)
    P3z = z6 * coefficients(P3)

    sol_x = P1z ./ P3z
    sol_y = P2z ./ P3z
    candidates = Vector{SMatrix{3, 3, Float64, 9}}(undef, length(sol_x))
    @inbounds @simd for i in 1:length(sol_x)
        candidates[i] = @. sol_x[i] * E1 + sol_y[i] * E2 + sol_z[i] * E3 + E4
    end
    candidates
end

function select_candidate(candidates, pd1, pd2; threshold, sample_ids)
    best_n_inliers = 0
    best_error = maxintfloat()
    best_inliers = fill(true, length(pd1))

    E_res = SMatrix{3, 3, Float64, 9}(I)
    inliers = fill(false, length(pd1))
    for E in candidates
        fill!(inliers, false)
        n_inliers, avg_error = compute_essential_error!(
            inliers, pd1, pd2, E, threshold)

        n_inliers < 5 && continue
        n_inliers == best_n_inliers && best_error ≤ avg_error && continue

        best_error = avg_error
        best_n_inliers = n_inliers
        copy!(best_inliers, inliers)
        E_res = E
    end

    best_n_inliers, (E_res, best_inliers, best_error)
end

"""
Test candidates for the essential matrix and return the best candidate.
"""
function select_candidates(
    candidates, pixels1, pixels2, K1, K2, cache::GEEV4x4Cache;
    max_repr_error = 1.0, sample_ids,
)
    best_n_inliers = 0
    best_repr_error = maxintfloat()
    best_inliers = fill(false, length(pixels1))

    inliers = fill(true, length(pixels1))
    sample_inliers = fill(true, length(sample_ids))
    @inbounds sample_px1, sample_px2 = pixels1[sample_ids], pixels2[sample_ids]

    E_res = SMatrix{3, 3, Float64, 9}(I)
    P_res = SMatrix{3, 4, Float64, 12}(I)
    P_ref = SMatrix{3, 4, Float64, 12}(I)

    for E in candidates
        for P in compute_projections(E)
            sample_n_inliers, _ = chirality_test!(
                inliers, sample_px1, sample_px2,
                P_ref, P, K1, K2, cache; max_repr_error)
            sample_n_inliers < length(sample_ids) && continue

            fill!(inliers, true)
            n_inliers, repr_error = chirality_test!(
                inliers, pixels1, pixels2, P_ref, P,
                K1, K2, cache; max_repr_error)

            n_inliers < 5 && continue
            n_inliers == best_n_inliers &&
                best_repr_error ≤ repr_error && continue

            best_repr_error = repr_error
            best_n_inliers = n_inliers
            copy!(best_inliers, inliers)
            E_res = E
            P_res = P
        end
    end
    best_n_inliers, (E_res, P_res, best_inliers, best_repr_error)
end

@inline recover_pose(E, pixels1, pixels2, K1, K2; threshold = 1.0) =
    recover_pose(E, pixels1, pixels2, K1, K2, GEEV4x4Cache(); threshold)

function recover_pose(
    E, pixels1, pixels2, K1, K2, cache::GEEV4x4Cache; threshold = 1.0,
)
    best_n_inliers = 0
    best_error = maxintfloat()
    best_inliers = fill(true, length(pixels1))
    inliers = fill(true, length(pixels1))

    P_res = SMatrix{3, 4, Float64, 12}(I)
    P_ref = SMatrix{3, 4, Float64, 12}(I)

    for P in compute_projections(E)
        fill!(inliers, true)
        n_inliers, repr_error = chirality_test!(
            inliers, pixels1, pixels2, P_ref, P, K1, K2, cache;
            max_repr_error=threshold)

        n_inliers < 5 && continue
        n_inliers == best_n_inliers && threshold ≤ repr_error && continue

        best_error = repr_error
        best_n_inliers = n_inliers
        copy!(best_inliers, inliers)
        P_res = P
    end
    best_n_inliers, P_res, best_inliers, best_error
end
