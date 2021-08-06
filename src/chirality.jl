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
function triangulate_point(p1, p2, P1, P2)
    x1, y1 = p1
    x2, y2 = p2
    A = SMatrix{4, 4, Float64}(
        x1*P1[3,1]-P1[1,1], y1*P1[3,1]-P1[2,1], x2*P2[3,1]-P2[1,1], y2*P2[3,1]-P2[2,1],
        x1*P1[3,2]-P1[1,2], y1*P1[3,2]-P1[2,2], x2*P2[3,2]-P2[1,2], y2*P2[3,2]-P2[2,2],
        x1*P1[3,3]-P1[1,3], y1*P1[3,3]-P1[2,3], x2*P2[3,3]-P2[1,3], y2*P2[3,3]-P2[2,3],
        x1*P1[3,4]-P1[1,4], y1*P1[3,4]-P1[2,4], x2*P2[3,4]-P2[1,4], y2*P2[3,4]-P2[2,4],
    )
    svd(A; full=true).V[:, 4]
end

"""
p1 & p2 in x, y format, pre-divided by K
"""
function chirality_test(p1, p2, P1, P2)
    n_points = length(p1)
    n_inliers = 0
    inliers = fill(false, n_points)

    for i in 1:n_points
        pt3d = triangulate_point(p1[i], p2[i], P1, P2)
        x1 = P1 * pt3d
        x2 = P2 * pt3d
        s = pt3d[4] < 0 ? -1 : 1
        s1 = x1[3] < 0 ? -1 : 1
        s2 = x2[3] < 0 ? -1 : 1
        if s * (s1 + s2) == 2
            n_inliers += 1
            inliers[i] = true
        end
        # If there are only 5 points, solution must be perfect, otherwise fail.
        # In other cases, there must be at least 75% inliers
        # for the solution to be considered valid.
        if (n_points == 5 && n_inliers < i) || (n_inliers < i / 4)
            return 0, inliers
        end
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
