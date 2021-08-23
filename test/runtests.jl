using Test
using LinearAlgebra
using Random
using StaticArrays

using RecoverPose

Random.seed!(0)

include("triangulate.jl")
include("p3p.jl")

function get_random_transform()
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
    t = SVector{3}(((rand(2) .- 0.5) ./ 10)..., 1)
    R, t
end

function get_target_E(R, t)
    # Compute essential matrix with R & t.
    tc = SMatrix{3, 3, Float64}(
        0, t[3], -t[2],
        -t[3], 0, t[1],
        t[2], -t[1], 0,
    )
    E = tc * R
    E /= E[3, 3]
end

@testset "Perfect Solution" begin
    K = SMatrix{3, 3}(I)
    R, t = get_random_transform()
    E_target = get_target_E(R, t)

    # Projection matrix.
    P_target = RecoverPose.get_transformation(R, t)
    P1 = K * RecoverPose.get_transformation(SMatrix{3, 3}(I), zeros(SVector{3}))
    P2 = K * P_target

    for n_points in (5, 50, 500)
        # Generate random points in `(x, y, z, w)` format.
        x = rand(4, n_points)
        x[4, :] .= 1

        x1h = P1 * x
        x2h = P2 * x
        x1h ./= reshape(x1h[3, :], (1, n_points))
        x2h ./= reshape(x2h[3, :], (1, n_points))
        x1 = x1h[1:2, :]
        x2 = x2h[1:2, :]

        # Convert to `(y, x)` format.
        x1 = [SVector{2}(x1[2, i], x1[1, i]) for i in 1:n_points]
        x2 = [SVector{2}(x2[2, i], x2[1, i]) for i in 1:n_points]

        if n_points == 5
            n_inliers, (E_res, P_res, inliers) = five_point(x1, x2, K, K)
            E = E_res ./ E_res[3, 3]
            @test all(isapprox.(E, E_target; atol=1e-3))
            @test all(isapprox.(P_res, P_target; atol=1e-3))
            @test sum(inliers) == n_points
            @test n_inliers == n_points
        end

        # Test 5pt RANSAC.

        n_inliers, (E_res, P_res, inliers) = five_point_ransac(x1, x2, K, K)
        E = E_res ./ E_res[3, 3]
        @test all(isapprox.(E, E_target; atol=1e-3))
        @test all(isapprox.(P_res, P_target; atol=1e-3))
        @test sum(inliers) == n_points
        @test n_inliers == n_points
    end
end
