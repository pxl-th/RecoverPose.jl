function backproject_transform(pixels, K_inv, R, t)
    points = Vector{SVector{3, Float64}}(undef, length(pixels))
    for (i, px) in enumerate(pixels)
        p = K_inv * SVector{3, Float64}(px[1], px[2], 1)
        p = R * p + t
        points[i] = p
    end
    points
end

@testset "P3P with minor noise." begin
    n_points = 10
    noise_scale = 1e-6
    
    fcmin, fcmax = 1e-3, 100
    K = SMatrix{3, 3, Float64}(
        rand() * (fcmax - fcmin) + fcmin, 0, 0,
        0, rand() * (fcmax - fcmin) + fcmin, 0,
        rand() * (fcmax - fcmin) + fcmin, rand() * (fcmax - fcmin) + fcmin, 1,
    )
    K_inv = K |> inv

    R = SMatrix{3, 3, Float64}(I)
    t = SVector{3, Float64}(1, 2, 3)

    max_res = 2 * min(K[1, 3], K[2, 3])
    pmin = SVector{2}(1, 1)
    pmax = SVector{2}(max_res, max_res)
    δ = pmax - pmin

    pixels = [floor.(rand(SVector{2, Float64}) .* δ .+ pmin) for i in 1:n_points]
    pdn_pixels = FivePoint.pre_divide_normalize(pixels, K)
    points = backproject_transform(pixels, K_inv, R, t)
    points = [p .+ rand(SVector{3, Float64}) .* 1e-6 for p in points]

    models = p3p(points[1:3], pdn_pixels[1:3], K)
    n_inliers, (P, inliers, error) = FivePoint.reprojection_error(
        models, pixels, points,
    )

    @test n_inliers == n_points
    @test sum(inliers) == n_inliers
    @test all(isapprox.(
        K_inv * P, FivePoint.get_transformation(R', -t); atol=1e-3,
    ))
end
