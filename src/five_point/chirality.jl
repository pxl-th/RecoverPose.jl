"""
Triangulate point given its two projection coordinates and projection matrices.

# Arguments
- `p1`: Pixel coordinates of a point in `(x, y)` format in the first view.
- `p2`: Pixel coordinates of a point in `(x, y)` format in the second view.
- `P1`: Projection matrix for the first view.
- `P2`: Projection matrix for the second view.

# Returns:
    Triangulated point in `(x, y, z, 1)` format.
"""
function triangulate_point(p1, p2, P1, P2; kwargs...)
    svd(_triangulation_system(p1, p2, P1, P2; kwargs...); full=true).V[:, 4]
end

function iterative_triangulation(p1, p2, P1, P2; ϵ::Real = 1e-6)
    ω1, ω2 = 1.0, 1.0
    res = SVector{4, Float64}(0, 0, 0, 0)
    total_iterations = 1
    for i in 1:10
        t = triangulate_point(p1, p2, P1, P2; ω1, ω2)
        t_norm = t / t[4]
        # Recalculate weights.
        ω1_new = P1[3, :] ⋅ t_norm
        ω2_new = P2[3, :] ⋅ t_norm
        abs(ω1_new - ω1) ≤ ϵ && abs(ω2_new - ω2) ≤ ϵ && (res = t; break)

        ω1, ω2 = ω1_new, ω2_new
        total_iterations += 1
    end
    inlier = ω1 > 0 && ω2 > 0 && total_iterations ≤ 10
    res, inlier
end

@inline function _triangulation_system(p1, p2, P1, P2; ω1 = 1.0, ω2 = 1.0)
    ω1, ω2 = 1.0 / ω1, 1.0 / ω2
    x1, y1 = p1
    x2, y2 = p2
    SMatrix{4, 4, Float64}(
        (x1 * P1[3,1] - P1[1,1]) * ω1, (y1 * P1[3,1] - P1[2,1]) * ω1, (x2 * P2[3,1] - P2[1,1]) * ω2, (y2 * P2[3,1] - P2[2,1]) * ω2,
        (x1 * P1[3,2] - P1[1,2]) * ω1, (y1 * P1[3,2] - P1[2,2]) * ω1, (x2 * P2[3,2] - P2[1,2]) * ω2, (y2 * P2[3,2] - P2[2,2]) * ω2,
        (x1 * P1[3,3] - P1[1,3]) * ω1, (y1 * P1[3,3] - P1[2,3]) * ω1, (x2 * P2[3,3] - P2[1,3]) * ω2, (y2 * P2[3,3] - P2[2,3]) * ω2,
        (x1 * P1[3,4] - P1[1,4]) * ω1, (y1 * P1[3,4] - P1[2,4]) * ω1, (x2 * P2[3,4] - P2[1,4]) * ω2, (y2 * P2[3,4] - P2[2,4]) * ω2,
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

        pt3d, inlier = iterative_triangulation(p1[i], p2[i], P1, P2)
        inlier || continue

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
