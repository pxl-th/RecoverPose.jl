"""
```julia
p3p(points, pdn_pixels::AbstractVector{SVector{3, T}}, K)
```

Recover pose using P3P algorithm.

# Arguments:

- `points: Vector of 3D points in `(x, y, z)` format.
- `pdn_pixels::AbstractVector{SVector{3, Float64}}`:
    Corresponding projections of `points` onto image plane,
    predivided by `K` intrinsics and normalized.
    E.g.: `Ki = inv(K); p = Ki [x, y, 1]; p /= norm(p)`.
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
    points, pdn_pixels::AbstractVector{SVector{3, T}}, K,
) where T <: Real
    models = SMatrix{3, 4, Float64}[]
    ϵ = 1e-4

    d12 = norm(points[1] .- points[2])
    d13 = norm(points[1] .- points[3])
    d23 = norm(points[2] .- points[3])
    (d12 < ϵ || d13 < ϵ || d23 < ϵ) && return models

    sd12, sd13, sd23 = d12^2, d13^2, d23^2

    α12 = pdn_pixels[1] ⋅ pdn_pixels[2]
    α13 = pdn_pixels[1] ⋅ pdn_pixels[3]
    α23 = pdn_pixels[2] ⋅ pdn_pixels[3]

    # Compute coefficients of a polynomial [7.80].
    m1, m2 = sd12, sd13 - sd23
    p1 = PPolynomial{Float64, :x}([0.0, -2.0 * α23 * sd12])
    p2 = PPolynomial{Float64, :x}([2.0 * α13 * sd23, - 2.0 * α23 * sd13])
    q1 = sd23 * PPolynomial{Float64, :x}([1.0, - 2.0 * α12, 1.0]) +
        PPolynomial{Float64, :x}([0.0, 0.0, -sd12])
    q2 = PPolynomial{Float64, :x}([sd23, 0.0, -sd13])

    P::PPolynomial{Float64, :x} =
        (m1 * q2 - m2 * q1)^2 - (m1 * p2 - m2 * p1) * (q1 * p2 - q2 * p1)
    P_roots::Vector{ComplexF64} = P |> Polynomials.roots

    for η12r in P_roots
        isreal(η12r) || continue
        η12 = real(η12r)
        η12 ≤ 0 && continue
        # Evaluate [7.89] at η12 to get η13.
        η13 = (m1 * q2 - m2 * q1)(η12) / (m1 * p2 - m2 * p1)(η12)
        # [7.91 - 7.93].
        denom = 1.0 + η12^2 - 2.0 * η12 * α12
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

        nx1 = η1 * pdn_pixels[1]
        z2 = η2 * pdn_pixels[2] - nx1; z2 /= norm(z2)
        z3 = η3 * pdn_pixels[3] - nx1; z3 /= norm(z3)
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

"""
```julia
p3p_select_model(models, points, pixels; threshold = 1.0)
```

Select best pose from `models`.

# Arguments:

- `models`: Projection matrices from which to select best one.
- `points`: 3D points in `(x, y, z)`
- `pixels`: Corresponding projections onto image plane in `(x, y)` format.
- `threshold`: Maximum distance in pixels between projected point
    and its target pixel, for the point to be considered inlier.
    Default value is `1.0`.

# Returns:

`n_inliers, (projection, inliers, error)`.

- `n_inliers`: Number of inliers for the `projection`.
- `projection`: `K * P` projection matrix that projects `points`
    onto image plane.
- `inliers`: Boolean vector indicating which point is inlier.
- `error`: Average reprojection error for the `pose`.
"""
function p3p_select_model(models, points, pixels; threshold = 1.0)
    best_n_inliers = 0
    best_error = maxintfloat()
    best_projection = nothing
    best_inliers = nothing

    for P in models
        avg_error = 0.0
        inliers = fill(false, length(points))
        n_inliers = 0
        for (i, (pixel, point)) in enumerate(zip(pixels, points))
            projected = P * SVector{4}(point..., 1.0)
            projected = projected[[1, 2]] ./ projected[3]
            error = norm(pixel .- projected)

            avg_error += error
            error ≥ threshold && continue

            inliers[i] = true
            n_inliers += 1
        end
        avg_error /= length(points)
        n_inliers < 3 && continue

        if avg_error < best_error
            best_projection = P
            best_inliers = inliers
            best_n_inliers = n_inliers
            best_error = avg_error
        end
    end

    best_n_inliers, (best_projection, best_inliers, best_error)
end

"""
```julia
pre_divide_normalize(pixels, K)
```

Divide `pixels` by `K` intrinsic matrix and normalize them.
`pixels` in `(x, y)` format.
"""
function pre_divide_normalize(pixels, K)
    res = Vector{SVector{3, Float64}}(undef, length(pixels))
    @inbounds for (i, px) in enumerate(pixels)
        p = SVector{3, Float64}(
            (px[1] - K[1, 3]) / K[1, 1], (px[2] - K[2, 3]) / K[2, 2], 1)
        res[i] = normalize(p)
    end
    res
end

function p3p(points, pixels::AbstractVector{SVector{2, T}}, K) where T <: Real
    p3p(points, pre_divide_normalize(pixels, K), K)
end

"""
```julia
p3p_ransac(points, pixels, pdn_pixels, K; threshold = 1.0, ransac_kwargs...)
```

Recover pose `K*[R|t]` using P3P Ransac algorithm.

# Arguments:

- `points`: 3D points in `(x, y, z)`
- `pixels`: Corresponding projections onto image plane in `(x, y)` format.
- `pdn_pixels`: Corresponding projections onto image plane,
    predivided by `K` intrinsics and normalized.
    E.g.: `Ki = inv(K); p = Ki [x, y, 1]; p /= norm(p)`.
- `K`: Camera intrinsics.
- `threshold`: Maximum distance in pixels between projected point
    and its target pixel, for the point to be considered inlier.
    Default value is `1.0`.
- `ransac_kwargs...`: Keyword arguments passed to [`ransac`](@ref).

# Returns:

`n_inliers, (projection, inliers, error)`.

- `n_inliers`: Number of inliers for the `projection`.
- `projection`: `K * P` projection matrix that projects `points`
    onto image plane.
- `inliers`: Boolean vector indicating which point is inlier.
- `error`: Average reprojection error for the `pose`.

# References:

```
Link: https://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
chapter: 7.3 Calibrated camera pose computation.
pages: 51-59
```
"""
function p3p_ransac(
    points, pixels, pdn_pixels, K; threshold = 1.0, ransac_kwargs...,
)
    sample_selection(sample_ids) = (points[sample_ids], pdn_pixels[sample_ids], K)
    rank(models; sample_ids) = p3p_select_model(models, points, pixels; threshold)
    ransac(sample_selection, p3p, rank, length(points), 3; ransac_kwargs...)
end

"""
```julia
p3p_ransac(points, pixels, K; threshold = 1.0, ransac_kwargs...)
```

Recover pose `K*[R|t]` using P3P Ransac algorithm.

# Arguments:

- `points`: 3D points in `(x, y, z)`
- `pixels`: Corresponding projections onto image plane in `(x, y)` format.
    These values, **WILL** be predivided and normalized by the intrinsic matrix.
    E.g.: `Ki = inv(K); p = Ki [x, y, 1]; p /= norm(p)`.
- `K`: Camera intrinsics.
- `threshold::Real`:
    Maximum distance in pixels between projected point and its target pixel,
    for the point to be considered inlier. Default value is `1.0`.
- `ransac_kwargs...`: Keyword arguments passed to [`ransac`](@ref).

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
function p3p_ransac(points, pixels, K; threshold = 1.0, ransac_kwargs...)
    pdn_pixels = pre_divide_normalize(pixels, K)
    p3p_ransac(points, pixels, pdn_pixels, K; threshold, ransac_kwargs...)
end
