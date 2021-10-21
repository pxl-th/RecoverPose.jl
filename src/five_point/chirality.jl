"""
```julia
triangulate(p1, p2, P1, P2)
```

Triangulate point given its two projection coordinates and projection matrices.

# Arguments:

- `p1`: Pixel coordinates of a point in `(x, y)` format in the first view.
- `p2`: Pixel coordinates of a point in `(x, y)` format in the second view.
- `P1`: `3x4` or `4x4` projection matrix `K * P` for the first view.
- `P2`: `3x4` or `4x4` projection matrix `K * P` for the second view.

# Returns:

Triangulated point in `(x, y, z, w)` format.
To get actual coordinates, divide by `w`.
"""
function triangulate(p1, p2, P1, P2, cache::GEEV4x4Cache)
    p, P = (p1, p2), (P1, P2)
    @inbounds A = SMatrix{4, 4, Float64, 16}(p[i][j] * P[i][3, k] - P[i][j, k] for j ∈ 1:2, i ∈ 1:2, k ∈ 1:4)
    copy!(cache.A, A' * A)
    __geev_4x4!(cache)
    @inbounds SVector{4, Float64}(@view(cache.VL[:, argmin(cache.WR)]))
end

function triangulate(p1, p2, P1, P2)
    p, P = (p1, p2), (P1, P2)
    @inbounds A = SMatrix{4, 4, Float64, 16}(p[i][j] * P[i][3, k] - P[i][j, k] for j ∈ 1:2, i ∈ 1:2, k ∈ 1:4)
    cache = GEEV4x4Cache()
    copy!(cache.A, A' * A)
    __geev_4x4!(cache)
    @inbounds SVector{4, Float64}(@view(cache.VL[:, argmin(cache.WR)]))
end

chirality_test!(inliers, points1, points2, P1, P2, K1, K2; max_repr_error = 1.0) =
    chirality_test!(inliers, points1, points2, P1, P2, K1, K2, GEEV4x4Cache(); max_repr_error)

function chirality_test!(
    inliers, pixels1, pixels2, P1, P2, K1, K2, cache::GEEV4x4Cache;
    max_repr_error = 1.0,
)
    n_points = length(pixels1)
    Pr1, Pr2 = K1 * P1, K2 * P2

    repr_error = 0.0
    n_inliers = 0
    xy_ids = SVector{2, UInt8}(1, 2)

    @inbounds for i ∈ 1:n_points
        pixel1, pixel2 = pixels1[i], pixels2[i]
        pt3d = triangulate(pixel1, pixel2, Pr1, Pr2, cache)
        if (pt3d[3] < 0 && pt3d[4] > 0) || (pt3d[3] > 0 && pt3d[4] < 0)
            inliers[i] = false
            continue
        end

        pt3d *= 1.0 / pt3d[4]
        if pt3d[3] ≥ 50
            inliers[i] = false
            continue
        end

        x2 = P2 * pt3d
        if !(0 < x2[3] < 50)
            inliers[i] = false
            continue
        end

        projection = Pr2 * pt3d
        pixel = projection[xy_ids] .* (1.0 / projection[3])
        Δ = pixel2 .- pixel
        error = √(Δ[1]^2 + Δ[2]^2)
        if error > max_repr_error
            inliers[i] = false
            continue
        end

        repr_error += error
        n_inliers += 1
    end
    repr_error /= n_inliers
    n_inliers, repr_error
end

function compute_essential_error!(inliers, p1, p2, E, threshold)
    threshold *= threshold
    n_inliers = 0
    avg_error = 0.0
    Et = E'

    @inbounds for i ∈ 1:length(p1)
        p1i = SVector{3, Float64}(p1[i]..., 1.0)
        p2i = SVector{3, Float64}(p2[i]..., 1.0)
        Ep1 = E * p1i
        Ep2 = Et * p2i

        error = p2i ⋅ Ep1
        error = (error * error) * (
            (1.0 / Ep1[1] * Ep1[1] + Ep1[2] * Ep1[2]) +
            (1.0 / Ep2[1] * Ep2[1] + Ep2[2] * Ep2[2]))
        if error < threshold
            inliers[i] = true
            n_inliers += 1
            avg_error += error
        end
    end
    avg_error /= n_inliers
    @show n_inliers, avg_error
    n_inliers, avg_error
end

function compute_projections(E)
    W = SMatrix{3, 3, Float64, 9}(0, 1, 0, -1, 0, 0, 0, 0, 1)

    F = svd(E; full=true)
    U, Vt = F.U, F.Vt
    det(U) < 0 && (U *= -1;)
    det(Vt) < 0 && (Vt *= -1;)

    t = @inbounds U[:, 3]
    t2 = -t
    R1 = U * W * Vt
    R2 = U * W' * Vt

    P1 = get_transformation(R1, t)
    P2 = get_transformation(R1, t2)
    P3 = get_transformation(R2, t)
    P4 = get_transformation(R2, t2)

    P1, P2, P3, P4
end
