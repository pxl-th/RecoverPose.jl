using Test
using LinearAlgebra
using Random
using StaticArrays
using Rotations
using BenchmarkTools

using RecoverPose

Random.seed!(0)

# include("triangulate.jl")
# include("p3p.jl")

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

@testset "Five Point: Perfect Solution" begin
    res = 1024
    pp = res ÷ 2
    K = SMatrix{3, 3, Float64}(
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

    x1, x2 = SVector{2, Float64}[], SVector{2, Float64}[]

    n_points = 1000
    for i in 1:n_points
        p1 = SVector{3, Float64}(rand() * res, rand() * res, 1.0)
        p1h = SVector{4, Float64}((K_inv * p1)..., 1.0)
        p2 = P2 * p1h
        p2 = p2[1:2] .* (1 / p2[3])

        push!(x1, p1[[1, 2]])
        push!(x2, p2[[1, 2]])
    end

    cache = GEEV4x4Cache()
    n_inliers, model = five_point_ransac(x1, x2, K, K, cache)
    E_res, P_res, inliers, repr_error = model
    E = E_res ./ E_res[3, 3]
    @test repr_error < 1e-5
    @test all(isapprox.(E, E_target; atol=1e-1))
    @test all(isapprox.(P_res[1:3, 1:3], P_target[1:3, 1:3]; atol=1e-1))
    @test norm(P_res[1:3, 4] .- P_target[1:3, 4]) < 1.0
    @test sum(inliers) == n_points
    @test n_inliers == n_points

    # n_inliers, model = essential_ransac(x1, x2, K, K)
    # E_res, inliers, e_error = model
    # E = E_res ./ E_res[3, 3]

    # _, P_res, _, _ = recover_pose(E_res, x1, x2, K, K)
    # display(P_target); println()
    # display(P_res); println()

    # @test e_error < 1e-5
    # @test all(isapprox.(E, E_target; atol=1e-1))
    # @test all(isapprox.(P_res[1:3, 1:3], P_target[1:3, 1:3]; atol=1e-1))
    # @test norm(P_res[1:3, 4] .- P_target[1:3, 4]) < 1.0
    # @test sum(inliers) == n_points
    # @test n_inliers == n_points

    @btime five_point_ransac($x1, $x2, $K, $K, $cache)
    # @btime essential_ransac($x1, $x2, $K, $K)
end
