"""
RANSAC algorithm used when there are more points than algorithm requires.

# Arguments:

- `sample_selector`: Function that given set of ids of the size `n_samples`,
    returns input for the `kernel` method.
- `kernel`: Function that calculates candidates from the sampled data.
- `rank`: Function that evaluates candidates, computed by `kernel`.
    It should return `n_inliers, model`, where `model` can be of any type.
    And is not used by the ransac itself. It only uses `n_inliers` to select
    which model is better.
- `n_points`: Total number of points in the data.
- `n_samples`: Number of points, that `kernel` function accepts.
    E.g. `five_point_ransac` sets this value to `5`.
- `iterations`: Maximum number of iterations to attempt. Default is `100`.
- `confidence`: Confidence in `[0, 1]` range. The higher the value, the more
    certain algorithm is that the picked model does not contain outliers.
    Number of iterations performed by RANSAC depends on this value as well.

# Returns:

Number of inliers and the selected model in the format returned by `kernel`.
"""
function ransac(
    sample_selector, kernel, rank, n_points, n_samples;
    iterations = 100, confidence = 0.99,
)
    best_M = nothing
    best_n_inliers = 0
    n = iterations

    ϵ = eps()
    current_iteration = 0
    while n > current_iteration
        ids = randperm(n_points)[1:n_samples]
        M = kernel(sample_selector(ids)...)
        n_inliers, M = rank(M; sample_ids=ids)

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
