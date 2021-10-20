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

const PPolynomial = Polynomials.Polynomial

include("ransac.jl")
include("five_point/utils.jl")
include("five_point/chirality.jl")
include("five_point/five_point.jl")
include("pnp/p3p.jl")

end
