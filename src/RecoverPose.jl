module RecoverPose
export five_point, five_point_ransac
export triangulate_point, iterative_triangulation
export p3p_select_model, p3p, p3p_ransac, pre_divide_normalize
export essential_ransac, recover_pose

using Statistics
using Random
using LinearAlgebra
using StaticArrays
using TypedPolynomials
using MultivariatePolynomials
using RowEchelon
import Polynomials

const PPolynomial = Polynomials.Polynomial

include("ransac.jl")

include("five_point/utils.jl")
include("five_point/chirality.jl")
include("five_point/five_point.jl")

include("pnp/p3p.jl")

precompile(iterative_triangulation, (
    SVector{2, Float64}, SVector{2, Float64},
    SMatrix{3, 4, Float64, 12}, SMatrix{3, 4, Float64, 12}))
precompile(iterative_triangulation, (
    SVector{2, Float64}, SVector{2, Float64},
    SMatrix{4, 4, Float64, 16}, SMatrix{4, 4, Float64, 16}))

precompile(five_point_candidates, (
    Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}))
precompile(select_candidates, (
    Vector{SMatrix{3, 3, Float64, 9}},
    Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}},
    SMatrix{3, 3, Float64, 9}, SMatrix{3, 3, Float64, 9}))
precompile(five_point, (
    Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}},
    SMatrix{3, 3, Float64, 9}, SMatrix{3, 3, Float64, 9}))
precompile(five_point_ransac, (
    Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}},
    SMatrix{3, 3, Float64, 9}, SMatrix{3, 3, Float64, 9}))

end
