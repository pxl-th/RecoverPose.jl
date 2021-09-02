function pre_divide(points1, points2, K1, K2)
    p1 = Vector{SVector{2, Float64}}(undef, length(points1))
    p2 = Vector{SVector{2, Float64}}(undef, length(points1))
    for i in 1:length(points1)
        p1[i] = SVector{2, Float64}(
            (points1[i][1] - K1[1, 3]) / K1[1, 1],
            (points1[i][2] - K1[2, 3]) / K1[2, 2],
        )
        p2[i] = SVector{2, Float64}(
            (points2[i][1] - K2[1, 3]) / K2[1, 1],
            (points2[i][2] - K2[2, 3]) / K2[2, 2],
        )
    end
    p1, p2
end

"""
- `points1`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the first image.
- `points2`:
    Pixel coordinates of the matched points in `(x, y)` format
    in the second image.
"""
function five_point(points1, points2, K1, K2)
    p1, p2 = pre_divide(points1, points2, K1, K2)
    candidates = five_point_candidates(p1, p2)
    select_candidates(candidates, points1, points2, p1, p2, K1, K2)
end

function five_point_ransac(points1, points2, K1, K2; ransac_kwargs...)
    p1, p2 = pre_divide(points1, points2, K1, K2)
    sample_selection(sample_ids) = (p1[sample_ids], p2[sample_ids])
    rank(models) = select_candidates(models, points1, points2, p1, p2, K1, K2)
    ransac(
        sample_selection, five_point_candidates, rank,
        length(p1), 5; ransac_kwargs...
    )
end

"""
Compute essential matrix using Five-Point algorithm.

# Arguments:
- `p1`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.
- `p2`: Vector of points, pre-divided by `K` matrix in `(x, y)` format.

# Returns:
    Essential matrix, projection matrix and boolean vector indicating inliers.
"""
function five_point_candidates(p1, p2)
    V = null_space(p1, p2)

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

    rref_M = compute_rref(E1, E2, E3, E4)

    eq_k = subtract(@view(rref_M[5, 11:20]), @view(rref_M[6, 11:20]))
    eq_l = subtract(@view(rref_M[7, 11:20]), @view(rref_M[8, 11:20]))
    eq_m = subtract(@view(rref_M[9, 11:20]), @view(rref_M[10, 11:20]))

    # Factorization.
    B11 = to_polynom(@view(eq_k[1:4]), z)
    B12 = to_polynom(@view(eq_k[5:8]), z)
    B13 = to_polynom(@view(eq_k[9:13]), z)

    B21 = to_polynom(@view(eq_l[1:4]), z)
    B22 = to_polynom(@view(eq_l[5:8]), z)
    B23 = to_polynom(@view(eq_l[9:13]), z)

    B31 = to_polynom(@view(eq_m[1:4]), z)
    B32 = to_polynom(@view(eq_m[5:8]), z)
    B33 = to_polynom(@view(eq_m[9:13]), z)

    # Calculate determinant.
    P1 = B23 * B12 - B13 * B22
    P2 = B13 * B21 - B23 * B11
    P3 = B11 * B22 - B12 * B21
    det_B = P1 * B31 + P2 * B32 + P3 * B33

    # Normalize the coefficient of the highest order.
    if (coefficient(det_B, z^10) ≉ 0)
        det_B = det_B / coefficient(det_B, z^10)
    end

    # Extract real roots of polynomial with companion matrix.
    coeffs = -coefficients(det_B)[end:-1:2]
    sol_z = eigen(vcat(hcat(zeros(9, 1), Matrix(I, 9, 9)), coeffs')).values
    sol_z = [real(s) for s in sol_z if isreal(s)]

    # Compute x & y.
    z6 = hcat(
        sol_z.^6, sol_z.^5, sol_z.^4, sol_z.^3, sol_z.^2, sol_z,
        ones(length(sol_z), 1),
    )
    z7 = hcat(sol_z.^7, z6)

    P1z = z7 * coefficients(P1)
    P2z = z7 * coefficients(P2)
    P3z = z6 * coefficients(P3)

    sol_x = P1z ./ P3z
    sol_y = P2z ./ P3z

    [@. sx*E1+sy*E2+sz*E3+E4 for (sx,sy,sz) in zip(sol_x,sol_y,sol_z)]
end

"""
Test candidates for the essential matrix and return the best candidate.
"""
function select_candidates(candidates, points1, points2, pdn1, pdn2, K1, K2)
    best_n_inliers = 0
    best_score = maxintfloat()
    best_repr_error = maxintfloat()
    best_inliers = Bool[]
    E_res = SMatrix{3, 3, Float64}(I)
    P_res = SMatrix{3, 4, Float64}(I)

    P_ref = SMatrix{3, 4, Float64}(I)
    for E in candidates
        for P in compute_projections(E)
            inliers = fill(true, length(points1))
            n_inliers, repr_error, inliers = chirality_test!(
                inliers,
                points1, points2, pdn1, pdn2,
                P_ref, P, K1, K2,
            )

            n_inliers < 5 && continue
            n_inliers == best_n_inliers &&
                best_repr_error ≤ repr_error && continue

            score = (length(points1) / n_inliers) * repr_error
            score ≥ best_score && continue

            best_score = score
            best_repr_error = repr_error
            best_n_inliers = n_inliers
            best_inliers = inliers
            E_res = E
            P_res = P
        end
    end
    best_n_inliers, (E_res, P_res, best_inliers, best_repr_error)
end
