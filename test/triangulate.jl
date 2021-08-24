@testset "Triangulation: " begin
    P1 = SMatrix{3, 4, Float64}(
        0.999701, -0.0171452, 0.0174497,
        0.0174497, 0.999695, -0.0174524,
        -0.017145, 0.0177517, 0.999695,
        -500, -100, -100,
    )
    P2 = SMatrix{3, 4, Float64}(
        0.99969, 0.0177543, -0.0174497,
        -0.0174497, 0.999695, 0.0174524,
        0.0177543, -0.0171425, 0.999695,
        500, -100, -100,
    )
    K = SMatrix{3, 3, Float64}(
        7291.67, 0, 0,
        0, 7291.67, 0,
        639.5, 511.5, 1,
    )

    P1 = K * P1
    P2 = K * P2

    p1 = SVector{2, Float64}(146, 642.288)
    p2 = SVector{2, Float64}(1137.31, 385.201)

    t = triangulate_point(p1, p2, P1, P2)
    t /= t[4]
    @test isapprox(t, SVector{4, Float64}(0, 100, 10000, 1); atol=1e-1)

    t = iterative_triangulation(p1, p2, P1, P2)
    t /= t[4]
    @test isapprox(t, SVector{4, Float64}(0, 100, 10000, 1); atol=1e-1)
end

@testset "Triangulation: Horizontal setup" begin
    P1 = SMatrix{3, 4, Float64}(I)
    P2 = hcat(SMatrix{3, 3, Float64}(I), SVector{3, Float64}(-1000, 0, 0))
    K = SMatrix{3, 3, Float64}(
        7291.67, 0, 0,
        0, 7291.67, 0,
        639.5, 511.5, 1,
    )

    P1 = K * P1
    P2 = K * P2

    p1 = SVector{2, Float64}(1004.08, 511.5)
    p2 = SVector{2, Float64}(274.917, 511.5)

    t = triangulate_point(p1, p2, P1, P2)
    t /= t[4]
    @test isapprox(t, SVector{4, Float64}(500, 0, 10000, 1); atol=1e-1)

    t = iterative_triangulation(p1, p2, P1, P2)
    t /= t[4]
    @test isapprox(t, SVector{4, Float64}(500, 0, 10000, 1); atol=1e-1)
end
