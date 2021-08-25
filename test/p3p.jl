function backproject_transform(pixels, K_inv, R, t)
    points = Vector{SVector{3, Float64}}(undef, length(pixels))
    for (i, px) in enumerate(pixels)
        p = K_inv * SVector{3, Float64}(px..., 1)
        points[i] = R * p + t
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

    θ = π / 8
    R = RotXYZ(rand() * θ, rand() * θ, rand() * θ)
    t = SVector{3, Float64}(rand(), 0, 0)
    P_target = RecoverPose.get_transformation(R', -t)

    max_res = 2 * min(K[1, 3], K[2, 3])
    pmin = SVector{2}(1, 1)
    pmax = SVector{2}(max_res, max_res)
    δ = pmax - pmin

    pixels = [floor.(rand(SVector{2, Float64}) .* δ .+ pmin) for i in 1:n_points]
    pdn_pixels = pre_divide_normalize(pixels, K)
    points = backproject_transform(pixels, K_inv, R, t)
    points = [p .+ rand(SVector{3, Float64}) .* noise_scale for p in points]

    models = p3p(points[1:3], pdn_pixels[1:3], K)
    n_inliers, (KP, inliers, error) = p3p_select_model(models, points, pixels; threshold=0.1)
    @test n_inliers == n_points
    @test sum(inliers) == n_inliers
    @test all(isapprox.(K_inv * KP, P_target; atol=1e-2))

    models = p3p(points[1:3], pixels[1:3], K)
    n_inliers, (KP, inliers, error) = p3p_select_model(models, points, pixels)
    @test n_inliers == n_points
    @test sum(inliers) == n_inliers
    @test all(isapprox.(K_inv * KP, P_target; atol=1e-2))
    
    # Test P3P RANSAC.

    n_inliers, (KP, inliers, error) = p3p_ransac(points, pixels, pdn_pixels, K)
    @test n_inliers == n_points
    @test sum(inliers) == n_inliers
    @test all(isapprox.(K_inv * KP, P_target; atol=1e-2))

    n_inliers, (KP, inliers, error) = p3p_ransac(points, pixels, K)
    @test n_inliers == n_points
    @test sum(inliers) == n_inliers
    @test all(isapprox.(K_inv * KP, P_target; atol=1e-2))
end

@testset "P3P RANSAC with noise and outliers." begin
    n_points = 50
    noise_scale = 1e-3
    rot_atol = 0.5
    t_atol = 2
    
    fcmin, fcmax = 1e-3, 100
    K = SMatrix{3, 3, Float64}(
        rand() * (fcmax - fcmin) + fcmin, 0, 0,
        0, rand() * (fcmax - fcmin) + fcmin, 0,
        rand() * (fcmax - fcmin) + fcmin, rand() * (fcmax - fcmin) + fcmin, 1,
    )
    K_inv = K |> inv

    R = SMatrix{3, 3, Float64}(I)
    t = SVector{3, Float64}(1, 2, 3)
    P_target = RecoverPose.get_transformation(R', -t)

    max_res = 2 * min(K[1, 3], K[2, 3])
    pmin = SVector{2}(1, 1)
    pmax = SVector{2}(max_res, max_res)
    δ = pmax - pmin

    pixels = [floor.(rand(SVector{2, Float64}) .* δ .+ pmin) for i in 1:n_points]
    pdn_pixels = pre_divide_normalize(pixels, K)
    points = backproject_transform(pixels, K_inv, R, t)
    points = [p .+ rand(SVector{3, Float64}) .* noise_scale for p in points]

    # Add outliers.
    outlier_scale = 5
    points[1] = points[1] .+ rand() * outlier_scale
    points[5] = points[5] .+ rand() * outlier_scale
    points[end] = points[end] .+ rand() * outlier_scale
    points[end - 5] = points[end - 5] .+ rand() * outlier_scale
    points[n_points ÷ 2] = points[n_points ÷ 2] .+ rand() * outlier_scale

    n_inliers, (KP, inliers, error) = p3p_ransac(points, pixels, pdn_pixels, K)
    P = K_inv * KP
    @test n_inliers ≥ n_points - 10
    @test !inliers[1] && !inliers[5] && !inliers[end] && !inliers[end - 5] &&
        !inliers[n_points ÷ 2]

    @test all(isapprox.(P[1:3, 1:3], P_target[1:3, 1:3]; atol=rot_atol))
    @test all(isapprox.(P[1:3, 4], P_target[1:3, 4]; atol=t_atol))

    n_inliers, (KP, inliers, error) = p3p_ransac(points, pixels, K)
    P = K_inv * KP
    @test n_inliers ≥ n_points - 10
    @test !inliers[1] && !inliers[5] && !inliers[end] && !inliers[end - 5] &&
        !inliers[n_points ÷ 2]

    @test all(isapprox.(P[1:3, 1:3], P_target[1:3, 1:3]; atol=rot_atol))
    @test all(isapprox.(P[1:3, 4], P_target[1:3, 4]; atol=t_atol))
end
