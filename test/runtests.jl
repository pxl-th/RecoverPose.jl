using Test
using LinearAlgebra
using Random
using StaticArrays
using Rotations

using RecoverPose

Random.seed!(0)

include("triangulate.jl")
include("p3p.jl")

"""
Compute essential matrix with R & t.
"""
function get_target_E(R, t)
    tc = SMatrix{3, 3, Float64}(
        0, t[3], -t[2],
        -t[3], 0, t[1],
        t[2], -t[1], 0,
    )
    E = tc * R
    E /= E[3, 3]
end

@testset "Perfect Solution" begin
    res = 1024
    pp = res ÷ 2
    K = SMatrix{3, 3}(
        910, 0, 0,
        0, 910, 0,
        pp, pp, 1,
    )
    K_inv = inv(K)

    θ = π / 8
    R = RotXYZ(rand() * θ, rand() * θ, rand() * θ)
    t = SVector{3, Float64}(rand(), rand() * 2, rand() * 3)

    P_target = RecoverPose.get_transformation(R, t)
    E_target = get_target_E(R, t)
    P2 = K * P_target

    for n_points in (5, 50, 500, 1000)
        x1, x2 = SVector{2, Float64}[], SVector{2, Float64}[]
        # Generate points for the both views.
        for i in 1:n_points
            p1 = SVector{3, Float64}(rand() * res, rand() * res, 1.0)
            p1h = SVector{4, Float64}((K_inv * p1)..., 1.0)
            p2 = P2 * p1h
            p2 = p2[1:2] .* (1 / p2[3])

            push!(x1, p1[[1, 2]])
            push!(x2, p2[[1, 2]])
        end

        if n_points == 5
            n_inliers, (E_res, P_res, inliers, repr_error) = five_point(x1, x2, K, K)
            E = E_res ./ E_res[3, 3]

            @test repr_error < 1e-5
            @test all(isapprox.(E, E_target; atol=1e-1))
            @test all(isapprox.(P_res[1:3, 1:3], P_target[1:3, 1:3]; atol=1e-1))
            @test norm(P_res[1:3, 4] .- P_target[1:3, 4]) < 0.5
            @test sum(inliers) == n_points
            @test n_inliers == n_points
        end

        # Test 5pt RANSAC.
        n_inliers, (E_res, P_res, inliers, repr_error) = five_point_ransac(x1, x2, K, K)
        E = E_res ./ E_res[3, 3]

        @test repr_error < 1e-5
        @test all(isapprox.(E, E_target; atol=1e-1))
        @test all(isapprox.(P_res[1:3, 1:3], P_target[1:3, 1:3]; atol=1e-1))
        @test norm(P_res[1:3, 4] .- P_target[1:3, 4]) < 0.5
        @test sum(inliers) == n_points
        @test n_inliers == n_points
    end
end
