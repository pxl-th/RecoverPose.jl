module RecoverPose
export GEEV4x4Cache
export five_point, five_point_ransac, essential_ransac, recover_pose
export p3p_select_model, p3p, p3p_ransac, pre_divide_normalize
export triangulate, ransac

using Statistics
using Random
using LinearAlgebra
using StaticArrays
using TypedPolynomials
using MultivariatePolynomials
using RowEchelon
import Polynomials

struct GEEV4x4Cache
    A::Matrix{Float64}
    VL::Matrix{Float64}
    VR::Matrix{Float64}
    WR::Vector{Float64}
    WI::Vector{Float64}
    work::Vector{Float64}
end

function GEEV4x4Cache()
    A = Matrix{Float64}(undef, 4, 4)
    VL = Matrix{Float64}(undef, 4, 4)
    VR = Matrix{Float64}(undef, 0, 0)
    WR = Vector{Float64}(undef, 4)
    WI = Vector{Float64}(undef, 4)
    work = Vector{Float64}(undef, 520)
    GEEV4x4Cache(A, VL, VR, WR, WI, work)
end

@inline function __geev_4x4!(cache::GEEV4x4Cache)
    n, A_stride = 4, 4
    jobvl, jobvr = 'V', 'N'
    lwork = LAPACK.BlasInt(520)
    info = Ref{LAPACK.BlasInt}()
    ccall(
        (:dgeev_64_, LAPACK.liblapack), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LAPACK.BlasInt}, Ptr{Float64},
         Ref{LAPACK.BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ref{LAPACK.BlasInt}, Ptr{Float64}, Ref{LAPACK.BlasInt}, Ptr{Float64},
         Ref{LAPACK.BlasInt}, Ptr{LAPACK.BlasInt}, Clong, Clong),
        jobvl, jobvr, n, cache.A, A_stride,
        cache.WR, cache.WI, cache.VL, n, cache.VR, n,
        cache.work, lwork, info, 1, 1)
end

using Rotations
using BenchmarkTools

const PPolynomial = Polynomials.Polynomial

include("ransac.jl")
include("five_point/utils.jl")
include("five_point/chirality.jl")
include("five_point/five_point.jl")
include("pnp/p3p.jl")

function main()
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

    x1, x2, K, K
end

function tt()
    T = Float64
    P1 = SMatrix{3, 4, T}(
        0.999701, -0.0171452, 0.0174497,
        0.0174497, 0.999695, -0.0174524,
        -0.017145, 0.0177517, 0.999695,
        -500, -100, -100,
    )
    P2 = SMatrix{3, 4, T}(
        0.99969, 0.0177543, -0.0174497,
        -0.0174497, 0.999695, 0.0174524,
        0.0177543, -0.0171425, 0.999695,
        500, -100, -100,
    )
    K = SMatrix{3, 3, T}(
        7291.67, 0, 0,
        0, 7291.67, 0,
        639.5, 511.5, 1,
    )

    P1 = K * P1
    P2 = K * P2

    p1 = SVector{2, T}(146, 642.288)
    p2 = SVector{2, T}(1137.31, 385.201)

    cache = GEEV4x4Cache()
    t = triangulate(p1, p2, P1, P2, cache)
    @show t / t[4]
    t = triangulate(p1, p2, P1, P2)
    @show t / t[4]

    @btime triangulate($p1, $p2, $P1, $P2, $cache)
    @btime triangulate($p1, $p2, $P1, $P2)
    @btime for _ in 1:1000_000 triangulate($p1, $p2, $P1, $P2, $cache) end
    @btime for _ in 1:1000_000 triangulate($p1, $p2, $P1, $P2) end

    nothing
end

end
