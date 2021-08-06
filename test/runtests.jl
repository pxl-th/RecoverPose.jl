using Test
using LinearAlgebra
using StaticArrays
using FivePoint
using Random

Random.seed!(0)

@testset "Identity K. Random transformation." begin
    K = SMatrix{3, 3}(I)
    # Rotation angle in radians.
    θ = (rand() - 0.5) * π * 360 / 180
    # Rotation axis.
    axis = rand(SVector{3, Float64})
    axis /= norm(axis)
    # Cross-product of rotation vector.
    v = SMatrix{3, 3}(
        0, axis[3], -axis[2],
        -axis[3], 0, axis[1],
        axis[2], -axis[1], 0,
    )
    # Rotation matrix (Rodrigues formula).
    R = SMatrix{3, 3}(I) .+ sin(θ) .* v .+ (1 - cos(θ)) .* (v * v)
    # Translation vector.
    t = SVector{3}(((rand(2) .- 0.5) ./ 10)..., 1)

    # Compute essential matrix with R & t.
    tc = SMatrix{3, 3, Float64}(
        0, t[3], -t[2],
        -t[3], 0, t[1],
        t[2], -t[1], 0,
    )
    E_target = tc * R
    E_target /= E_target[3, 3]

    # Projection matrix.
    P_target = FivePoint.get_transformation(R, t)
    p1 = K * FivePoint.get_transformation(SMatrix{3, 3}(I), zeros(SVector{3}))
    p2 = K * P_target

    # Generate random points in `(x, y, z, w)` format.
    n_points = 5
    x = rand(4, n_points)
    x[4, :] .= 1

    x1h = p1 * x
    x2h = p2 * x
    x1h ./= reshape(x1h[3, :], (1, n_points))
    x2h ./= reshape(x2h[3, :], (1, n_points))
    x1 = x1h[1:2, :]
    x2 = x2h[1:2, :]

    # Convert to `(y, x)` format.
    x1 = [SVector{2}(x1[2, i], x1[1, i]) for i in 1:n_points]
    x2 = [SVector{2}(x2[2, i], x2[1, i]) for i in 1:n_points]

    E_res, P_res, inliers = five_point(x1, x2, K, K)
    E_res /= E_res[3, 3]

    @test all(isapprox.(P_res, P_target; atol=1e-2))
    @test all(E_res .≈ E_target)
    @test sum(inliers) == n_points
end
