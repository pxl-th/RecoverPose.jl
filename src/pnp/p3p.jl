"""
pixels: K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
"""
function p3p(points, pixels, K)
    ϵ = 1e-4

    d12 = norm(points[1] .- points[2])
    d13 = norm(points[1] .- points[3])
    d23 = norm(points[2] .- points[3])
    (d12 < ϵ || d13 < ϵ || d23 < ϵ) && return nothing

    sd12, sd13, sd23 = d12^2, d13^2, d23^2

    α12 = pixels[1] ⋅ pixels[2]
    α13 = pixels[1] ⋅ pixels[3]
    α23 = pixels[2] ⋅ pixels[3]

    # Compute coefficients of a polynomial [7.80].
    m1, m2 = sd12, sd13 - sd23
    p1 = PPolynomial([0, -2 * α23 * sd12])
    p2 = PPolynomial([2 * α13 * sd23, - 2 * α23 * sd13])
    q1 = sd23 * PPolynomial([1, - 2 * α12, 1]) + PPolynomial([0, 0, -sd12])
    q2 = PPolynomial([sd23, 0, -sd13])

    P = (m1 * q2 - m2 * q1)^2 - (m1 * p2 - m2 * p1) * (q1 * p2 - q2 * p1)
    P_roots = P |> Polynomials.roots

    models = SMatrix{3, 4, Float64}[]

    for η12 in P_roots
        η12 ≤ 0 && continue
        # Evaluate [7.89] at η12 to get η13.
        η13 = (m1 * q2 - m2 * q1)(η12) / (m1 * p2 - m2 * p1)(η12)
        # [7.91 - 7.93].
        denom = 1 + η12^2 - 2 * η12 * α12
        denom ≤ 0 && continue
        η1 = d12 / √denom
        η2 = η1 * η12
        η3 = η1 * η13
        (η1 ≤ 0 || η2 ≤ 0 || η3 ≤ 0) && continue
        # Compute and check errors.
        (
            abs((√(η1^2 + η2^2 - 2 * η1 * η2 * α12) - d12) / d12) > ϵ ||
            abs((√(η1^2 + η3^2 - 2 * η1 * η3 * α13) - d13) / d13) > ϵ ||
            abs((√(η2^2 + η3^2 - 2 * η2 * η3 * α23) - d23) / d23) > ϵ
        ) && continue

        nx1 = η1 * pixels[1]
        z2 = η2 * pixels[2] - nx1; z2 /= norm(z2)
        z3 = η3 * pixels[3] - nx1; z3 /= norm(z3)
        z1 = z2 × z3; z1 /= norm(z1)
        z3z1 = z3 × z1

        zw2 = @. (points[2] - points[1]) / d12
        zw3 = @. (points[3] - points[1]) / d31
        zw1 = zw2 × zw3; zw1 /= norm(zw1)
        zw3zw1 = zw3 × zw1

        # Recover rotation R [7.130 - 7.134].
        Z = SMatrix{3, 3, Float64}(z1..., z2..., z3z1...)
        ZW = SMatrix{3, 3, Float64}(zw1..., zw2..., zw3zw1...)
        # TODO check if this is correct or should:
        # https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/calib3d/src/usac/utils.cpp#L138
        R = Z * int(ZW)
        KR = K * R
        t = -KR * (points[1] - R' * nx1)
        push!(models, SMatrix{3, 4, Float64}(KR..., t...))
    end
    models
end

Random.seed!(0)

function main()
    n = 3
    
    pmin, pmax = SVector{3}(-1, -1, 5), SVector{3}(1, 1, 10)
    point_cloud = [rand(SVector{3, Float64}) .* (pmax .- pmin) .+ pmin for _ in 1:n]
    
    fcmin, fcmax = 1e-3, 100
    K = SMatrix{3, 3, Float64}(
        rand() * (fcmax - fcmin) + fcmin, 0, 0,
        0, rand() * (fcmax - fcmin) + fcmin, 0,
        rand() * (fcmax - fcmin) + fcmin, rand() * (fcmax - fcmin) + fcmin, 1,
    )

    @info "K"
    display(K); println()

    models = p3p(point_cloud, point_cloud, K)
    @info models
end

main()
