get_transformation(R, t) = SMatrix{3, 4, Float64}(R..., t...)

"""
For polynom, with coefficients as matrices, compute traces of the coefficients.
"""
trace(p) = mapreduce(mi -> tr(coefficient.(p, mi)) * mi, +, monomials(p[1]))

function to_polynom(coeffs, x)
    n = length(coeffs)
    sum(coeffs[i] * x ^ (n - i) for i in 1:n)
end

function null_space(p1, p2)
    n = length(p1)
    F = Matrix{Float64}(undef, n, 9)
    @inbounds for i in 1:n
        F[i, 1] = p2[i][1] * p1[i][1] # x2 * x1
        F[i, 2] = p2[i][1] * p1[i][2] # x2 * y1
        F[i, 3] = p2[i][1]            # x2

        F[i, 4] = p2[i][2] * p1[i][1] # y2 * x1
        F[i, 5] = p2[i][2] * p1[i][2] # y2 * y1
        F[i, 6] = p2[i][2]            # y2

        F[i, 7] = p1[i][1]            # x1
        F[i, 8] = p1[i][2]            # y1
        F[i, 9] = 1
    end
    svd(F; full=true).V
end

"""
input -- 10-element vector
"""
function subtract(v1, v2)
    v1 = [0, v1[1:3]..., 0, v1[4:6]..., 0, v1[7:10]...]
    v2 = [v2[1:3]..., 0, v2[4:6]..., 0, v2[7:10]..., 0]
    v1 .- v2
end

function compute_rref(E1, E2, E3, E4)
    # One equation from rank constraint.
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

    row33 = mat_part .- 0.5 .* trace_part
    row9 = mapreduce(
        mi -> reshape(coefficient.(row33, mi), 9),
        hcat, monomials(row33[1]),
    )

    M = vcat(coefficients(row1)', row9) # 10x20 matrix.

    order = [1,7,2,4,3,11,8,14,5,12,6,13,17,9,15,18,10,16,19,20]
    Base.permutecols!!(M, order)
    rref(M)
end
