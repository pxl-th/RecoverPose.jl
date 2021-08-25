using Test
using LinearAlgebra
using Random
using StaticArrays
using Rotations

using RecoverPose

Random.seed!(0)

include("triangulate.jl")
include("p3p.jl")

function get_random_transform()
    R = RotXYZ(rand(), rand(), rand())
    t = SVector{3}(((rand(2) .- 0.5))..., 1.0)
    R, t
end

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

    R, t = get_random_transform()
    P_target = RecoverPose.get_transformation(R, t)
    E_target = get_target_E(R, t)
    P2 = K * P_target # Projection matrix.

    for n_points in (5, 50, 500, 1000)
        x1, x2 = SVector{2}[], SVector{2}[]
        for i in 1:n_points
            p1 = SVector{3}(rand() * res, rand() * res, 1.0)
            p1h = SVector{4}((K_inv * p1)..., 1.0)
            p2 = P2 * p1h
            p2 = p2[1:2] .* (1 / p2[3])

            push!(x1, p1[[1, 2]])
            push!(x2, p2[[1, 2]])
        end

        @info "N Points $n_points"
        if n_points == 5
            n_inliers, (E_res, P_res, inliers) = five_point(x1, x2, K, K)
            E = E_res ./ E_res[3, 3]

            @test all(isapprox.(E, E_target; atol=1e-1))
            @test all(isapprox.(P_res, P_target; atol=1e-1))
            @test sum(inliers) == n_points
            @test n_inliers == n_points
        end

        # Test 5pt RANSAC.
        n_inliers, (E_res, P_res, inliers) = five_point_ransac(x1, x2, K, K)
        E = E_res ./ E_res[3, 3]

        @test all(isapprox.(E, E_target; atol=1e-1))
        @test all(isapprox.(P_res, P_target; atol=1e-1))
        @test sum(inliers) == n_points
        @test n_inliers == n_points
    end
end
