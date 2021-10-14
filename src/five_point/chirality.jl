"""
```julia
triangulate(p1, p2, P1, P2)
```

Triangulate point given its two projection coordinates and projection matrices.

# Arguments
- `p1`: Pixel coordinates of a point in `(x, y)` format in the first view.
- `p2`: Pixel coordinates of a point in `(x, y)` format in the second view.
- `P1`: 3x4 Projection matrix for the first view.
- `P2`: 3x4 Projection matrix for the second view.

# Returns:
    Triangulated point in `(x, y, z, w)` format.
    To get actual coordinates, divide by `w`.
"""
function triangulate(p1, p2, P1, P2)::SVector{4, Float64}
    p, P = (p1, p2), (P1, P2)
    @inbounds A = SMatrix{4, 4, Float64, 16}(
            p[i][j] * P[i][3, k] - P[i][j, k] for j ∈ 1:2, i ∈ 1:2, k ∈ 1:4)
    eigen(A' * A).vectors[:, 1]
end

function iterative_triangulation(p1, p2, P1, P2; ϵ::Float64 = 1e-5)
    ω1, ω2 = 1.0, 1.0
    x = SVector{4, Float64}(0, 0, 0, 0)
    @inbounds for _ ∈ 1:10
        A = _triangulation_system(p1, p2, P1, P2, ω1, ω2)
        x = real(eigvecs(A' * A)[:, 1])
        ω1_new = P1[3, :] ⋅ x
        ω2_new = P2[3, :] ⋅ x

        (abs(ω1_new) < ϵ || abs(ω2_new) < ϵ) && break
        abs(ω1_new - ω1) ≤ ϵ && abs(ω2_new - ω2) ≤ ϵ && break
        ω1, ω2 = ω1_new, ω2_new
    end
    x
end

@inline function _triangulation_system(p1, p2, P1, P2, ω1, ω2)
    ω1, ω2 = 1.0 / ω1, 1.0 / ω2
    @inbounds begin
    c1 = (p1[1] .* P1[3, :] .- P1[1, :]) .* ω1
    c2 = (p1[2] .* P1[3, :] .- P1[2, :]) .* ω1
    c3 = (p2[1] .* P2[3, :] .- P2[1, :]) .* ω2
    c4 = (p2[2] .* P2[3, :] .- P2[2, :]) .* ω2
    m = SMatrix{4, 4, Float64, 16}(
        c1[1], c2[1], c3[1], c4[1],
        c1[2], c2[2], c3[2], c4[2],
        c1[3], c2[3], c3[3], c4[3],
        c1[4], c2[4], c3[4], c4[4])
    end
    m
end

function chirality_test!(
    inliers, points1, points2, P1, P2, K1, K2; max_repr_error = 1.0,
)
    n_points = length(points1)
    Pr1, Pr2 = K1 * P1, K2 * P2

    repr_error = 0.0
    n_inliers = 0
    xy_ids = SVector{2, UInt8}(1, 2)

    @inbounds for i ∈ 1:n_points
        point1, point2 = points1[i], points2[i]

        pt3d = triangulate(point1, point2, Pr1, Pr2)
        pt3d *= 1.0 / pt3d[4]
        if !(0 < pt3d[3] < 50)
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
        Δ = point2 .- pixel
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
        error = (error * error) / (
            Ep1[1] * Ep1[1] + Ep1[2] * Ep1[2] +
            Ep2[1] * Ep2[1] + Ep2[2] * Ep2[2])
        if error < threshold
            inliers[i] = true
            n_inliers += 1
            avg_error += error
        end
    end

    n_inliers, (avg_error / length(p1))
end

function compute_projections(E)
    W = SMatrix{3, 3, Float64, 9}(0, 1, 0, -1, 0, 0, 0, 0, 1)

    F = svd(E; full=true)
    U, Vt = F.U, F.Vt
    det(U) < 0 && (U *= -1;)
    det(Vt) < 0 && (Vt *= -1;)

    t = U[:, 3]
    t2 = -t
    R1 = U * W * Vt
    R2 = U * W' * Vt

    P1 = get_transformation(R1, t)
    P2 = get_transformation(R1, t2)
    P3 = get_transformation(R2, t)
    P4 = get_transformation(R2, t2)

    P1, P2, P3, P4
end
