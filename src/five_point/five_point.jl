function pre_divide(points1, points2, K1, K2)
    p1 = Vector{SVector{2, Float64}}(undef, length(points1))
    p2 = Vector{SVector{2, Float64}}(undef, length(points1))

    @inbounds fx1, fy1, px1, py1 = K1[1, 1], K1[2, 2], K1[1, 3], K1[2, 3]
    @inbounds fx2, fy2, px2, py2 = K2[1, 1], K2[2, 2], K2[1, 3], K2[2, 3]

    @inbounds @simd for i in 1:length(points1)
        p1[i] = SVector{2, Float64}(
            (points1[i][1] - K1[1, 3]) / K1[1, 1],
            (points1[i][2] - K1[2, 3]) / K1[2, 2])
        p2[i] = SVector{2, Float64}(
            (points2[i][1] - K2[1, 3]) / K2[1, 1],
            (points2[i][2] - K2[2, 3]) / K2[2, 2])
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
function five_point_ransac(
    points1, points2, K1, K2; max_repr_error = 1.0, ransac_kwargs...,
)
    p1, p2 = pre_divide(points1, points2, K1, K2)
    sample_selection(sample_ids) = (p1[sample_ids], p2[sample_ids])
    rank(models) = select_candidates(
        models, points1, points2, K1, K2; max_repr_error)
    ransac(
        sample_selection, five_point_candidates, rank,
        length(p1), 5; ransac_kwargs...)
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
    threshold /= (K1[1, 1] + K1[2, 2]) / 2.0

    pd1, pd2 = pre_divide(pixels1, pixels2, K1, K2)
    sample_selection(sample_ids) = (pd1[sample_ids], pd2[sample_ids])
    rank(models) = select_candidate(models, pd1, pd2; threshold)
    ransac(
        sample_selection, five_point_candidates, rank,
        length(pd1), 5; ransac_kwargs...)
end

"""
Compute essential matrix using Five-Point algorithm.

# Arguments:

- `p1`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.
- `p2`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.

# Returns:

Vector of `3x3` essential matrix candidates.
"""
function five_point_candidates(p1, p2)
    @polyvar z

    V = null_space(p1, p2)
    E1 = SMatrix{3, 3, Float64, 9}(V[:, 6,]...)'
    E2 = SMatrix{3, 3, Float64, 9}(V[:, 7]...)'
    E3 = SMatrix{3, 3, Float64, 9}(V[:, 8]...)'
    E4 = SMatrix{3, 3, Float64, 9}(V[:, 9]...)'
    rref_M = compute_rref(E1, E2, E3, E4)

    eq_k = subtract(@view(rref_M[5, 11:20]), @view(rref_M[6, 11:20]))
    eq_l = subtract(@view(rref_M[7, 11:20]), @view(rref_M[8, 11:20]))
    eq_m = subtract(@view(rref_M[9, 11:20]), @view(rref_M[10, 11:20]))

    # Factorization.
    v4, v5 = Val(4), Val(5)
    B11 = to_polynom(@view(eq_k[1:4]), z, v4)
    B12 = to_polynom(@view(eq_k[5:8]), z, v4)
    B13 = to_polynom(@view(eq_k[9:13]), z, v5)

    B21 = to_polynom(@view(eq_l[1:4]), z, v4)
    B22 = to_polynom(@view(eq_l[5:8]), z, v4)
    B23 = to_polynom(@view(eq_l[9:13]), z, v5)

    B31 = to_polynom(@view(eq_m[1:4]), z, v4)
    B32 = to_polynom(@view(eq_m[5:8]), z, v4)
    B33 = to_polynom(@view(eq_m[9:13]), z, v5)

    # Calculate determinant.
    P1 = B23 * B12 - B13 * B22
    P2 = B13 * B21 - B23 * B11
    P3 = B11 * B22 - B12 * B21
    det_B = P1 * B31 + P2 * B32 + P3 * B33

    # Normalize the coefficient of the highest order.
    (coefficient(det_B, z^10) ≉ 0) &&
        (det_B /= coefficient(det_B, z^10);)

    # Extract real roots of polynomial with companion matrix.
    Z = zeros(Float64, 10, 10)
    sub_Z = @view(Z[1:9, 2:10])
    sub_Z[diagind(sub_Z)] .= 1.0
    Z[10, :] .= -coefficients(det_B)[end:-1:2]
    sol_z = Float64[real(s) for s in eigvals!(Z) if isreal(s)]
    isempty(sol_z) && return SMatrix{3, 3, Float64, 9}[]

    # Compute x & y.
    z6 = Matrix{Float64}(undef, length(sol_z), 7)
    z7 = Matrix{Float64}(undef, length(sol_z), 8)
    z6[:, 7] .= 1.0
    z7[:, 8] .= 1.0
    z7[:, 1] .= sol_z.^7
    for i in 1:6
        v = sol_z .^ i
        z6[:, 7 - i] .= v
        z7[:, 8 - i] .= v
    end

    P1z = z7 * coefficients(P1)
    P2z = z7 * coefficients(P2)
    P3z = z6 * coefficients(P3)

    sol_x = P1z ./ P3z
    sol_y = P2z ./ P3z
    [@. sx*E1+sy*E2+sz*E3+E4 for (sx,sy,sz) in zip(sol_x,sol_y,sol_z)]
end

function select_candidate(candidates, pd1, pd2; threshold)
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
    candidates, points1, points2, K1, K2; max_repr_error = 1.0,
)
    best_n_inliers = 0
    best_repr_error = maxintfloat()
    best_inliers = fill(true, length(points1))
    inliers = fill(true, length(points1))

    E_res = SMatrix{3, 3, Float64, 9}(I)
    P_res = SMatrix{3, 4, Float64, 12}(I)
    P_ref = SMatrix{3, 4, Float64, 12}(I)

    for E in candidates
        for P in compute_projections(E)
            fill!(inliers, true)
            n_inliers, repr_error = chirality_test!(
                inliers, points1, points2, P_ref, P, K1, K2; max_repr_error)

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

function recover_pose(E, pixels1, pixels2, K1, K2; threshold = 1.0)
    best_n_inliers = 0
    best_error = maxintfloat()
    best_inliers = fill(true, length(pixels1))
    inliers = fill(true, length(pixels1))

    P_res = SMatrix{3, 4, Float64, 12}(I)
    P_ref = SMatrix{3, 4, Float64, 12}(I)

    for P in compute_projections(E)
        fill!(inliers, true)
        n_inliers, repr_error = chirality_test!(
            inliers, pixels1, pixels2, P_ref, P, K1, K2;
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

