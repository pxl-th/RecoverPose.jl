function ransac(
    sample_selector, kernel, rank, n_points::Int, n_samples::Int;
    iterations = 100, confidence = 0.99,
)
    ϵ = eps()

    best_M = nothing
    best_n_inliers = 0
    n = iterations

    current_iteration = 0
    while n > current_iteration
        ids = randperm(n_points)[1:n_samples]

        M = kernel(sample_selector(ids)...)
        n_inliers, M = rank(M)

        if n_inliers > best_n_inliers
            best_M = M
            best_n_inliers = n_inliers
            # Update `n` estimate, which is the number of trials to ensure
            # a dataset with no outliers is picked with given `confidence`.
            p_no_outliers = 1.0 - (n_inliers / n_points) ^ n_samples
            p_no_outliers = clamp(p_no_outliers, ϵ, 1 - ϵ)
            n = log(1 - confidence) / log(p_no_outliers)
        end

        current_iteration += 1
        current_iteration > iterations && break
    end

    best_n_inliers, best_M
end
