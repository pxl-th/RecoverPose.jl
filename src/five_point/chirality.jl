"""
Triangulate point given its two projection coordinates and projection matrices.

# Arguments
- `p1`: Pixel coordinates of a point in `(x, y)` format in the first view.
- `p2`: Pixel coordinates of a point in `(x, y)` format in the second view.
- `P1`: 3x4 Projection matrix for the first view.
- `P2`: 3x4 Projection matrix for the second view.

# Returns:
    Triangulated point in `(x, y, z, w)` format.
"""
function triangulate_point(p1, p2, P1, P2)
    A = _triangulation_system(p1, p2, P1, P2)
    V = eigvecs(A' * A)[:, 1]
end

function iterative_triangulation(p1, p2, P1, P2; ϵ::Float64 = 1e-5)
    ω1, ω2 = 1.0, 1.0
    x = SVector{4, Float64}(0, 0, 0, 0)
    for _ in 1:10
        A = _triangulation_system(p1, p2, P1, P2, ω1, ω2)
        x = real(eigvecs(A' * A)[:, 1])
        ω1_new = P1[3, :] ⋅ x
        ω2_new = P2[3, :] ⋅ x
        abs(ω1_new - ω1) ≤ ϵ && abs(ω2_new - ω2) ≤ ϵ && break

        ω1, ω2 = ω1_new, ω2_new
    end
    x
end

@inline function _triangulation_system(p1, p2, P1, P2, ω1, ω2)
    ω1, ω2 = 1.0 / ω1, 1.0 / ω2
    c1 = (p1[1] .* P1[3, :] .- P1[1, :]) .* ω1
    c2 = (p1[2] .* P1[3, :] .- P1[2, :]) .* ω1
    c3 = (p2[1] .* P2[3, :] .- P2[1, :]) .* ω2
    c4 = (p2[2] .* P2[3, :] .- P2[2, :]) .* ω2
    SMatrix{4, 4, Float64}(
        c1[1], c2[1], c3[1], c4[1],
        c1[2], c2[2], c3[2], c4[2],
        c1[3], c2[3], c3[3], c4[3],
        c1[4], c2[4], c3[4], c4[4],
    )
end

@inline function _triangulation_system(p1, p2, P1, P2)
    c1 = p1[1] .* P1[3, :] .- P1[1, :]
    c2 = p1[2] .* P1[3, :] .- P1[2, :]
    c3 = p2[1] .* P2[3, :] .- P2[1, :]
    c4 = p2[2] .* P2[3, :] .- P2[2, :]
    SMatrix{4, 4, Float64}(
        c1[1], c2[1], c3[1], c4[1],
        c1[2], c2[2], c3[2], c4[2],
        c1[3], c2[3], c3[3], c4[3],
        c1[4], c2[4], c3[4], c4[4],
    )
end

function chirality_test!(inliers, points1, points2, P1, P2, K1, K2)
    n_points = length(points1)
    Pr1, Pr2 = K1 * P1, K2 * P2

    repr_error = 0.0
    n_inliers = 0
    for i in 1:n_points
        pt3d = iterative_triangulation(points1[i], points2[i], Pr1, Pr2)
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

        pp2 = Pr2 * pt3d
        pp2 = pp2[1:2] ./ pp2[3]
        repr_error += norm(points2[i] .- pp2)
        n_inliers += 1
    end

    repr_error /= length(points1)
    n_inliers, repr_error, inliers
end

function compute_projections(E)
    W = SMatrix{3, 3, Float64}(
        0, 1, 0,
        -1, 0, 0,
        0, 0, 1,
    )

    F = svd(E; full=true)
    U, Vt = F.U, F.Vt
    det(U) < 0 && (U *= -1;)
    det(Vt) < 0 && (Vt *= -1;)

    t = U[:, 3]
    R1 = U * W * Vt
    R2 = U * W' * Vt

    P1 = get_transformation(R1, t)
    P2 = get_transformation(R1, -t)
    P3 = get_transformation(R2, t)
    P4 = get_transformation(R2, -t)

    P1, P2, P3, P4
end
