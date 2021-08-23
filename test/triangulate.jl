@testset "Triangulate" begin
    P1 = SMatrix{3, 4, Float64}(
        5010, 0, 0,
        0, 5010, 0,
        360, 640, 1,
        0, 0, 0,
    )
    P2 = SMatrix{3, 4, Float64}(
         5.037e+03, 2.148e+02, 3.925e-01,
        -9.611e+01, 5.354e+03, 7.092e-02,
        -1.756e+03, 1.918e+02, 9.169e-01,
         4.284e+03, 8.945e+02, 4.930e-01,
    )
    P3 = SMatrix{3, 4, Float64}(
        5.217e+03, -5.734e+02, -3.522e-01,
        5.217e+03, -5.734e+02, -3.522e-01,
        2.366e+03, 8.233e+02, 9.340e-01,
        -3.799e+03, -2.567e+02, 6.459e-01,
    )

    x1 = SVector{2, Float64}(274.128, 624.409)
    x2 = SVector{2, Float64}(239.571, 533.568)
    x3 = SVector{2, Float64}(297.574, 549.260)

    t = triangulate_point(x1, x2, P1, P2)
    t /= t[4]
    @info t
    t, inlier = iterative_triangulation(x1, x2, P1, P2)
    t /= t[4]
    @info t

    t = triangulate_point(x1, x3, P1, P3)
    t /= t[4]
    @info t
    t, inlier = iterative_triangulation(x1, x3, P1, P3)
    t /= t[4]
    @info t

    t = triangulate_point(x2, x3, P2, P3)
    t /= t[4]
    @info t
    t, inlier = iterative_triangulation(x2, x3, P2, P3)
    t /= t[4]
    @info t
end
