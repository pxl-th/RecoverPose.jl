# FivePoint.jl

Five-point algorithm for computing an essential matrix from a set of corresponding points.

### Install

```julia
]add https://github.com/pxl-th/FivePoint.jl.git
```

### Use

- Pass points along with intrinsics:

Pass `Vector{SVector{2}}` points for both views along with their intrinsic matrices.
**NOTE** that in this case points should be in `(y, x)` format.

```julia
E, P, inliers, n_inliers = five_point(points1, points2, K1, K2)
```

- Pre-divided by `K`:

Alternativelly you can pre-divide points by their intrinsic matrices.
**NOTE** that in this case points should be in `(x, y)` format.

```julia
E, P, inliers, n_inliers = five_point(points1, points2)
```

**Returns:**

- `E`: array of essential matrices;
- `P`: array of projection matrices;
- `inliers`: array of boolean vectors indicating what point is inlier;
- `n_inliers`: specifies number of inliers.

These arrays contain multiple values in case more than one solution has passed the chirality test
with the maximum amount of inliers. In this case, they usually have different projection matrices.

### References

```
D. Nister, "An efficient solution to the five-point relative pose problem,"
in IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 26, no. 6, pp. 756-770, June 2004, doi: 10.1109/TPAMI.2004.17.
```
