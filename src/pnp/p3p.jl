"""
Recover pose [R|t] using P3P algorithm.

# Arguments:
- `points::Vector{SVector{3, Float64}}`: 3D points in `(x, y, z)`
- `pixels::Vector{SVector{3, Float64}}`:
    Corresponding projections onto image plane,
    predivided by `K` intrinsics and normalized.
    E.g.: Ki = int(K); p = Ki [x, y, 1]; p /= norm(p)
- `K::SMatrix{3, 3, Float64}`: Camera intrinsics.

# Returns:
`Vector{SMatrix{3, 4, Float64}}` vector of up to 4 possible solutions.
Each element is a projection matrix `P = K * [R|t]`.
To get pure transformation matrix, multiply `P` by `inv(K)`.

# References:
```
Link: https://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
chapter: 7.3 Calibrated camera pose computation.
pages: 51-59
```
"""
function p3p(
    points::Vector{SVector{3, Float64}},
    pixels::Vector{SVector{3, Float64}},
    K::SMatrix{3, 3, Float64},
)
    models = SMatrix{3, 4, Float64}[]
    ϵ = 1e-4

    d12 = norm(points[1] .- points[2])
    d13 = norm(points[1] .- points[3])
    d23 = norm(points[2] .- points[3])
    (d12 < ϵ || d13 < ϵ || d23 < ϵ) && return models

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

    for η12 in P_roots
        isreal(η12) || continue
        η12 = η12 |> real
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
        zw3 = @. (points[3] - points[1]) / d13
        zw1 = zw2 × zw3; zw1 /= norm(zw1)
        zw3zw1 = zw3 × zw1

        # Recover rotation R [7.130 - 7.134].
        Z = SMatrix{3, 3, Float64}(z1..., z2..., z3z1...)
        ZW = SMatrix{3, 3, Float64}(zw1..., zw2..., zw3zw1...)
        R = Z * inv(ZW)
        KR = K * R
        Kt = -KR * (points[1] - R' * nx1)
        push!(models, SMatrix{3, 4, Float64}(KR..., Kt...))
    end
    models
end

function p3p_select_model(
    models::Vector{SMatrix{3, 4, Float64}},
    points::Vector{SVector{3, Float64}},
    pixels::Vector{SVector{2, T}};
    threshold::Real = 1.0,
) where T <: Real
    best_n_inliers = 0
    best_error = maxintfloat()
    best_projection = nothing
    best_inliers = nothing

    for P in models
        avg_error = 0.0
        inliers = fill(false, length(points))
        n_inliers = 0
        for (i, (pixel, point)) in enumerate(zip(pixels, points))
            projected = P[1:3, 1:3] * point + P[1:3, 4]
            projected = projected[1:2] ./ projected[3]
            error = norm(pixel .- projected)
            avg_error += error
            error < threshold && (inliers[i] = true; n_inliers += 1;)
        end
        avg_error /= length(points)

        if avg_error < best_error
            best_projection = P
            best_inliers = inliers
            best_n_inliers = n_inliers
            best_error = avg_error
        end
    end

    best_n_inliers, (best_projection, best_inliers, best_error)
end

function pre_divide_normalize(
    pixels::Vector{SVector{2, T}}, K::SMatrix{3, 3, Float64},
) where T <: Real
    res = Vector{SVector{3, Float64}}(undef, length(pixels))
    for (i, px) in enumerate(pixels)
        p = SVector{3}((px[1]-K[1,3])/K[1,1], (px[2]-K[2,3])/K[2,2],1)
        res[i] = normalize(p)
    end
    res
end

function p3p(
    points::Vector{SVector{3, Float64}},
    pixels::Vector{SVector{2, Float64}},
    K::SMatrix{3, 3, Float64},
)
    p3p(points, pre_divide_normalize(pixels, K), K)
end

"""
Recover pose `K*[R|t]` using P3P Ransac algorithm.

# Arguments:
- `points::Vector{SVector{3, Float64}}`: 3D points in `(x, y, z)`
- `pixels::Vector{SVector{2, Float64}}`:
    Corresponding projections onto image plane in `(x, y)` format.
    These values should be predivided and normalized by the intrinsic matrix.
    E.g.: Ki = int(K); p = Ki [x, y, 1]; p /= norm(p)
- `pdn_pixels::Vector{SVector{3, Float64}}`:
    Corresponding projections onto image plane,
    predivided by `K` intrinsics and normalized.
    E.g.: Ki = int(K); p = Ki [x, y, 1]; p /= norm(p)
- `K::SMatrix{3, 3, Float64}`: Camera intrinsics.
- `threshold::Real`:
    Maximum distance in pixels between projected point and its target pixel,
    for the point to be considered inlier. Default value is `1.0`.

# Returns:
`Vector{SMatrix{3, 4, Float64}}` vector of up to 4 possible solutions.
Each element is a projection matrix `P = K * [R|t]`.
To get pure transformation matrix, multiply `P` by `inv(K)`.

# References:
```
Link: https://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
chapter: 7.3 Calibrated camera pose computation.
pages: 51-59
```
"""
function p3p_ransac(
    points::Vector{SVector{3, Float64}},
    pixels::Vector{SVector{2, Float64}},
    pdn_pixels::Vector{SVector{3, Float64}},
    K::SMatrix{3, 3, Float64};
    threshold::Real = 1.0,
    ransac_kwargs...,
)
    sample_selection(sample_ids) =
        (points[sample_ids], pdn_pixels[sample_ids], K)
    rank(models) = p3p_select_model(models, points, pixels; threshold)
    ransac(
        sample_selection, p3p, rank,
        length(points), 3;
        ransac_kwargs...
    )
end

"""
Recover pose `K*[R|t]` using P3P Ransac algorithm.

# Arguments:
- `points::Vector{SVector{3, Float64}}`: 3D points in `(x, y, z)`
- `pixels::Vector{SVector{2, Float64}}`:
    Corresponding projections onto image plane in `(x, y)` format.
    These values, will be predivided and normalized by the intrinsic matrix.
    E.g.: Ki = int(K); p = Ki [x, y, 1]; p /= norm(p)
- `K::SMatrix{3, 3, Float64}`: Camera intrinsics.
- `threshold::Real`:
    Maximum distance in pixels between projected point and its target pixel,
    for the point to be considered inlier. Default value is `1.0`.

# Returns:
`Vector{SMatrix{3, 4, Float64}}` vector of up to 4 possible solutions.
Each element is a projection matrix `P = K * [R|t]`.
To get pure transformation matrix, multiply `P` by `inv(K)`.

# References:
```
Link: https://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
chapter: 7.3 Calibrated camera pose computation.
pages: 51-59
```
"""
function p3p_ransac(
    points::Vector{SVector{3, Float64}},
    pixels::Vector{SVector{2, Float64}},
    K::SMatrix{3, 3, Float64},
    threshold::Real = 1.0,
    ransac_kwargs...,
)
    pdn_pixels = pre_divide_normalize(pixels, K)
    p3p_ransac(points, pixels, pdn_pixels, K; threshold, ransac_kwargs...)
end
