function ransac(
    sample_selector, kernel, rank, n_points::Int, n_samples::Int;
    iterations = 100, confidence = 0.99,
)
    best_M = nothing
    best_n_inliers = 0
    n = iterations

    ϵ = eps()
    current_iteration = 0
    while n > current_iteration
        ids = randperm(n_points)[1:n_samples]
        t1 = time()
        M = kernel(sample_selector(ids)...)
        t2 = time()

        t1 = time()
        n_inliers, M = rank(M)
        t2 = time()

        if n_inliers > best_n_inliers
            best_M = M
            best_n_inliers = n_inliers
            # Update `n` estimate, which is the number of trials to ensure
            # a dataset with no outliers is picked with given `confidence`.
            p_no_outliers = 1.0 - (n_inliers / n_points) ^ n_samples
            p_no_outliers = clamp(p_no_outliers, ϵ, 1.0 - ϵ)
            n = log(1.0 - confidence) / log(p_no_outliers)
        end

        current_iteration += 1
        current_iteration > iterations && break
    end

    best_M ≡ nothing && return nothing
    best_n_inliers, best_M
end
