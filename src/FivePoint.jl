module FivePoint
export five_point

using LinearAlgebra
using StaticArrays
using TypedPolynomials
using MultivariatePolynomials
using RowEchelon

"""
- `points1`:
    Pixel coordinates of the matched points in `(y, x)` format
    in the first image.
- `points2`:
    Pixel coordinates of the matched points in `(y, x)` format
    in the second image.
"""
function five_point(points1, points2, K1, K2)
    (length(points1) != length(points2) || length(points1) < 5) && throw(
        "Number of points should be ≥ 5, " *
        "and both vectors should have the same amount of points. " *
        "Instead, number of points is $(length(points1)) & $(length(points2))."
    )

    # TODO move this to outside of the function.
    # Points are now in `(x, y)` format.
    p1 = Matrix{Float64}(undef, length(points1), 2)
    p2 = Matrix{Float64}(undef, length(points1), 2)
    # TODO get rid of either p1 of pp1 (same for p2).
    pp1 = Vector{SVector{2, Float64}}(undef, length(points1))
    pp2 = Vector{SVector{2, Float64}}(undef, length(points1))
    for i in 1:length(points1)
        pp1[i] = SVector{2, Float64}(
            (points1[i][2] - K1[1, 3]) / K1[1, 1],
            (points1[i][1] - K1[2, 3]) / K1[2, 2],
        )
        pp2[i] = SVector{2, Float64}(
            (points2[i][2] - K2[1, 3]) / K2[1, 1],
            (points2[i][1] - K2[2, 3]) / K2[2, 2],
        )
        p1[i, :] = pp1[i]
        p2[i, :] = pp2[i]
    end

    F = Matrix{Float64}(undef, length(points1), 9)
    for i in 1:length(points1)
        F[i, 1] = p2[i, 1] * p1[i, 1] # x2 * x1
        F[i, 2] = p2[i, 1] * p1[i, 2] # x2 * y1
        F[i, 3] = p2[i, 1]            # x2

        F[i, 4] = p2[i, 2] * p1[i, 1] # y2 * x1
        F[i, 5] = p2[i, 2] * p1[i, 2] # y2 * y1
        F[i, 6] = p2[i, 2]             # y2

        F[i, 7] = p1[i, 1]             # x1
        F[i, 8] = p1[i, 2]             # y1

        F[i, 9] = 1.0
    end

    # TODO do we need V or Vt?
    V = svd(F; full=true).V

    # TODO do we need to transpose?
    E1 = SMatrix{3, 3, Float64}(
        V[1, 6], V[4, 6], V[7, 6],
        V[2, 6], V[5, 6], V[8, 6],
        V[3, 6], V[6, 6], V[9, 6],
    )
    E2 = SMatrix{3, 3, Float64}(
        V[1, 7], V[4, 7], V[7, 7],
        V[2, 7], V[5, 7], V[8, 7],
        V[3, 7], V[6, 7], V[9, 7],
    )
    E3 = SMatrix{3, 3, Float64}(
        V[1, 8], V[4, 8], V[7, 8],
        V[2, 8], V[5, 8], V[8, 8],
        V[3, 8], V[6, 8], V[9, 8],
    )
    E4 = SMatrix{3, 3, Float64}(
        V[1, 9], V[4, 9], V[7, 9],
        V[2, 9], V[5, 9], V[8, 9],
        V[3, 9], V[6, 9], V[9, 9],
    )

    # One equation from rank constraint.
    @polyvar x y z
    # Coefficients.
    c11 = E1[1, 1] * x + E2[1, 1] * y + E3[1, 1] * z + E4[1, 1]
    c12 = E1[1, 2] * x + E2[1, 2] * y + E3[1, 2] * z + E4[1, 2]
    c13 = E1[1, 3] * x + E2[1, 3] * y + E3[1, 3] * z + E4[1, 3]

    c21 = E1[2, 1] * x + E2[2, 1] * y + E3[2, 1] * z + E4[2, 1]
    c22 = E1[2, 2] * x + E2[2, 2] * y + E3[2, 2] * z + E4[2, 2]
    c23 = E1[2, 3] * x + E2[2, 3] * y + E3[2, 3] * z + E4[2, 3]

    c31 = E1[3, 1] * x + E2[3, 1] * y + E3[3, 1] * z + E4[3, 1]
    c32 = E1[3, 2] * x + E2[3, 2] * y + E3[3, 2] * z + E4[3, 2]
    c33 = E1[3, 3] * x + E2[3, 3] * y + E3[3, 3] * z + E4[3, 3]

    # row1 - vector containing 20 coefficients for the first equation.
    row1 =
        c11 * c22 * c33 + c12 * c23 * c31 +
        c13 * c21 * c32 - c13 * c22 * c31 -
        c12 * c21 * c33 - c11 * c23 * c32

    # 9 equations from trace constraint.
    # Coefficients.
    e1 = E1 * x + E2 * y + E3 * z + E4
    e2 = E1' * x + E2' * y + E3' * z + E4'

    mat_part = (e1 * e2) * e1
    trace_part = trace(e1 * e2) * e1
    # TODO is this correct?
    row33 = mat_part - 0.5 * trace_part
    row9 = mapreduce(
        mi -> reshape(coefficient.(row33, mi), 9),
        hcat, monomials(row33[1]),
    )

    M = vcat(coefficients(row1)', row9) # 10x20 matrix.

    col_order = [
        1, 7, 2, 4, 3, 11, 8, 14, 5, 12, 6, 13, 17, 9, 15, 18, 10, 16, 19, 20,
    ]
    Base.permutecols!!(M, col_order)
    rref_M = rref(M)

    eq_k = subtr(rref_M[5, 11:20], rref_M[6, 11:20])
    eq_l = subtr(rref_M[7, 11:20], rref_M[8, 11:20])
    eq_m = subtr(rref_M[9, 11:20], rref_M[10, 11:20])

    # Factorization.
    B11 = eq_k[1] * z^3 + eq_k[2] * z^2 + eq_k[3] * z + eq_k[4]
    B12 = eq_k[5] * z^3 + eq_k[6] * z^2 + eq_k[7] * z + eq_k[8]
    B13 = eq_k[9] * z^4 + eq_k[10] * z^3 + eq_k[11] * z^2 + eq_k[12] * z + eq_k[13]
    B21 = eq_l[1] * z^3 + eq_l[2] * z^2 + eq_l[3] * z + eq_l[4]
    B22 = eq_l[5] * z^3 + eq_l[6] * z^2 + eq_l[7] * z + eq_l[8]
    B23 = eq_l[9] * z^4 + eq_l[10] * z^3 + eq_l[11] * z^2 + eq_l[12] * z + eq_l[13]
    B31 = eq_m[1] * z^3 + eq_m[2] * z^2 + eq_m[3] * z + eq_m[4]
    B32 = eq_m[5] * z^3 + eq_m[6] * z^2 + eq_m[7] * z + eq_m[8]
    B33 = eq_m[9] * z^4 + eq_m[10] * z^3 + eq_m[11] * z^2 + eq_m[12] * z + eq_m[13]
    # Calculate determinant.
    P1 = B23 * B12 - B13 * B22
    P2 = B13 * B21 - B23 * B11
    P3 = B11 * B22 - B12 * B21
    det_B = P1 * B32 + P2 * B32 + P3 * B33
    # Normalize the coefficient of the highest order.
    if (coefficient(det_B, z^10) ≉ 0)
        det_B = det_B / coefficient(det_B, z^10)
    end
    # Extract roots of polynomial with companion matrix.
    coeffs = -coefficients(det_B)[end:-1:2]
    sol_z = eigen(vcat(hcat(zeros(9, 1), Matrix(I, 9, 9)), coeffs')).values
    # Select only real roots.
    sol_z = [real(s) for s in sol_z if isreal(s)]
    # Compute x & y.
    z6 = hcat(
        sol_z .^ 6, sol_z .^ 5, sol_z .^ 4,
        sol_z .^ 3, sol_z .^ 2, sol_z, ones(length(sol_z), 1),
    )
    z7 = hcat(sol_z .^ 7, z6)

    P1z = z7 * coefficients(P1)
    P2z = z7 * coefficients(P2)
    P3z = z6 * coefficients(P3)

    sol_x = P1z ./ P3z
    sol_y = P2z ./ P3z

    # Perform chirality testing to select best E candidate.
    best_inliers = 0
    E_res = nothing
    P_res = nothing

    P_ref = SMatrix{4, 4, Float64}(I)
    for i in 1:length(sol_z)
        E = @. sol_x[i] * E1 + sol_y[i] * E2 + sol_z[i] * E3 + E4
        for P in compute_projections(E)
            n_inliers = chirality_test(pp1, pp2, P_ref, P)
            if n_inliers > best_inliers
                best_inliers = n_inliers
                E_res = E
                P_res = P
            end
        end
    end

    E_res, P_res, best_inliers
end

"""
p1 & p2 in x, y format, pre-divided by K
"""
function chirality_test(p1, p2, P1, P2)
    n_inliers = 0
    n_points = length(p1)
    for i in 1:n_points
        pt3d = triangulate_point(p1[i], p2[i], P1, P2)
        x1 = P1 * pt3d
        x2 = P2 * pt3d
        s = pt3d[4] < 0 ? -1 : 1
        s1 = x1[3] < 0 ? -1 : 1
        s2 = x2[3] < 0 ? -1 : 1
        if (s1 + s2) * s == 2
            n_inliers += 1
        end
        # If there are only 5 points, solution must be perfect, otherwise fail.
        # In other cases, there must be at least 75% inliers
        # for the solution to be considered valid.
        if (n_points == 5 && n_inliers < i) || (n_inliers < i / 4)
            return 0
        end
    end
    n_inliers
end

function compute_projections(E)
    W = SMatrix{3, 3, Float64}(
         0, 1, 0,
        -1, 0, 0,
         0, 0, 0,
    )
    F = svd(E; full=true)

    R1 = F.U * W * F.Vt
    R2 = F.U * W' * F.Vt
    t = F.U[:, 3]

    P1 = get_transformation(R1, t)
    P2 = get_transformation(R1, -t)
    P3 = get_transformation(R2, t)
    P4 = get_transformation(R2, -t)

    P1, P2, P3, P4
end

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

function get_transformation(R, t)
    SMatrix{3, 4, Float64}(
        R[1, 1], R[1, 2], R[1, 3], t[1],
        R[2, 1], R[2, 2], R[2, 3], t[2],
        R[3, 1], R[3, 2], R[3, 3], t[3],
    )
end

trace(p) = mapreduce(mi -> tr(coefficient.(p, mi)) * mi, +, monomials(p[1]))

"""
input -- 10-element vector
"""
function subtr(v1, v2)
    v1 = [0, v1[1:3]..., 0, v1[4:6]..., 0, v1[7:10]...]
    v2 = [v2[1:3]..., 0, v2[4:6]..., 0, v2[7:10]..., 0]
    v1 .- v2
end

end
