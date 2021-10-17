using Documenter
using RecoverPose

makedocs(
    sitename="RecoverPose.jl",
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/RecoverPose.jl/blob/{commit}{path}#L{line}",
    modules=[RecoverPose],
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Five Point" => "five-point.md",
            "P3P" => "p3p.md",
            "Triangulation" => "triangulation.md",
            "RANSAC" => "ransac.md"]
    ],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"))
deploydocs(;repo="github.com/pxl-th/RecoverPose.jl.git")
