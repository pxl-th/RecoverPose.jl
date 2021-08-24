# RecoverPose.jl

Different methods for pose recovery.

### Methods

- Five Point for Essential matrix.
- Five Point Ransac.
- P3P (Perspective-3-Point) for projection matrix.
- P3P Ransac.
- Eigen Triangulation.
- Iterative Eigen Triangulation.

### Install

```julia
]add https://github.com/pxl-th/RecoverPose.jl.git
```

### References

- Five point:
```
D. Nister, "An efficient solution to the five-point relative pose problem,"
in IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 26, no. 6, pp. 756-770, June 2004, doi: 10.1109/TPAMI.2004.17.
```

- P3P:
```
Link: https://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
chapter: 7.3 Calibrated camera pose computation.
pages: 51-59
```
