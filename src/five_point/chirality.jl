"""
Triangulate point given its two projection coordinates and projection matrices.

# Arguments
- `p1`: Pixel coordinates of a point in `(x, y)` format in the first view.
- `p2`: Pixel coordinates of a point in `(x, y)` format in the second view.
- `P1`: Projection matrix for the first view.
- `P2`: Projection matrix for the second view.

# Returns:
    Triangulated point in `(x, y, z, w)` format.
"""
function triangulate_point(p1, p2, P1, P2)
    A = _triangulation_system(p1, p2, P1, P2)
    V = eigvecs(A' * A)[:, 1]
end

function iterative_triangulation(p1, p2, P1, P2; ϵ::Real = 1e-5)
    ω1, ω2 = 1.0, 1.0
    x = SVector{4, Float64}(0, 0, 0, 0)
    for i in 1:10
        A = _triangulation_system(p1, p2, P1, P2)
        x = eigvecs(A' * A)[:, 1]
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

"""
p1 & p2 in x, y format, pre-divided by K
"""
function chirality_test(p1, p2, P1, P2, inliers)
    threshold = 50
    n_points = length(p1)
    is_exact = n_points == 5

    n_inliers = 0
    for i in 1:n_points
        inliers[i] || continue

        pt3d = iterative_triangulation(p1[i], p2[i], P1, P2)
        if (pt3d[3] > 0 && pt3d[4] > 0) || (pt3d[3] < 0 && pt3d[4] < 0)
            pt3d *= 1.0 / pt3d[4]
            if pt3d[3] < threshold
                x2 = P2 * pt3d
                (0 < x2[3] < threshold) && (inliers[i] = true; n_inliers += 1;)
            end
        end
        # If there are only 5 points, solution must be perfect, otherwise fail.
        is_exact && n_inliers < i && return 0, inliers
    end
    n_inliers, inliers
end

function compute_projections(E)
    W = SMatrix{3, 3, Float64}(
         0, 1, 0,
        -1, 0, 0,
         0, 0, 1,
    )
    F = svd(E; full=true)

    t = F.U[:, 3]
    R1 = F.U * W * F.Vt
    R2 = F.U * W' * F.Vt
    det(R1) < 0 && (R1 *= -1)
    det(R2) < 0 && (R2 *= -1)

    P1 = get_transformation(R1, t)
    P2 = get_transformation(R1, -t)
    P3 = get_transformation(R2, t)
    P4 = get_transformation(R2, -t)

    P1, P2, P3, P4
end
