@inline get_transformation(R, t) = SMatrix{3, 4, Float64, 12}(R..., t...)

"""
For polynom, with coefficients as matrices, compute traces of the coefficients.
"""
function trace(p::SMatrix{3, 3, T, 9})::T where T
    mapreduce(mi -> tr(coefficient.(p, mi)) * mi, +, monomials(p[1]))
end

function null_space!(F, p1, p2)
    @inbounds @simd for i in 1:length(p1)
        p1i, p2i = p1[i], p2[i]
        F[i, 1] = p2i[1] * p1i[1] # x2 * x1
        F[i, 2] = p2i[1] * p1i[2] # x2 * y1
        F[i, 3] = p2i[1]          # x2

        F[i, 4] = p2i[2] * p1i[1] # y2 * x1
        F[i, 5] = p2i[2] * p1i[2] # y2 * y1
        F[i, 6] = p2i[2]          # y2

        F[i, 7] = p1i[1]          # x1
        F[i, 8] = p1i[2]          # y1
        F[i, 9] = 1.0
    end
    svd(F; full=true).V
end

@inbounds function to_polynoms(v1, v2, z)
    p1 = -v2[1] * z^3 + (v1[1] - v2[2]) * z^2 + (v1[2] - v2[3]) * z + v1[3]
    p2 = -v2[4] * z^3 + (v1[4] - v2[5]) * z^2 + (v1[5] - v2[6]) * z + v1[6]
    p3 = -v2[7] * z^4 + (v1[7] - v2[8]) * z^3 + (v1[8] - v2[9]) * z^2 + (v1[9] - v2[10]) * z + v1[10]
    p1, p2, p3
end

function compute_rref!(M, E1, E2, E3, E4)
    @polyvar x y z

    # One equation from rank constraint.
    @inbounds c11 = E1[1, 1] * x + E2[1, 1] * y + E3[1, 1] * z + E4[1, 1]
    @inbounds c12 = E1[1, 2] * x + E2[1, 2] * y + E3[1, 2] * z + E4[1, 2]
    @inbounds c13 = E1[1, 3] * x + E2[1, 3] * y + E3[1, 3] * z + E4[1, 3]

    @inbounds c21 = E1[2, 1] * x + E2[2, 1] * y + E3[2, 1] * z + E4[2, 1]
    @inbounds c22 = E1[2, 2] * x + E2[2, 2] * y + E3[2, 2] * z + E4[2, 2]
    @inbounds c23 = E1[2, 3] * x + E2[2, 3] * y + E3[2, 3] * z + E4[2, 3]

    @inbounds c31 = E1[3, 1] * x + E2[3, 1] * y + E3[3, 1] * z + E4[3, 1]
    @inbounds c32 = E1[3, 2] * x + E2[3, 2] * y + E3[3, 2] * z + E4[3, 2]
    @inbounds c33 = E1[3, 3] * x + E2[3, 3] * y + E3[3, 3] * z + E4[3, 3]

    # row1 - vector containing 20 coefficients for the first equation.
    row1 =
        c11 * c22 * c33 + c12 * c23 * c31 +
        c13 * c21 * c32 - c13 * c22 * c31 -
        c12 * c21 * c33 - c11 * c23 * c32

    # 9 equations from trace constraint.
    e1 = E1 .* x + E2 .* y + E3 .* z + E4
    e2 = E1' .* x + E2' .* y + E3' .* z + E4'

    mat_part = (e1 * e2) * e1
    trace_part = trace(e1 * e2) * e1
    row33 = mat_part .- 0.5 .* trace_part

    @inbounds monoms = monomials(row33[1])
    @inbounds @simd for i in 1:length(monoms)
        M[1:9, i] .= reshape(coefficient.(row33, monoms[i]), 9)
    end
    @inbounds M[10, :] .= coefficients(row1)

    order = UInt8[1,7,2,4,3,11,8,14,5,12,6,13,17,9,15,18,10,16,19,20]
    Base.permutecols!!(M, order)
    rref!(M)
end
