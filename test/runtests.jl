using Test
using StaticArrays
using FivePoint

@testset "Test" begin
    points1 = [
        SVector{2}(1, 1),
        SVector{2}(2, 2),
        SVector{2}(3, 3),
        SVector{2}(4, 4),
        SVector{2}(5, 5),
    ]
    points2 = [
        SVector{2}(2, 2),
        SVector{2}(3, 3),
        SVector{2}(4, 4),
        SVector{2}(5, 5),
        SVector{2}(6, 6),
    ]
    K = SMatrix{3, 3, Float64}(
        910, 0, 0,
        0, 910, 0,
        64, 64, 1,
    )

    E, P, n_inliers = five_point(points1, points2, K, K)
    @show n_inliers
    @show E
    @show P
end
