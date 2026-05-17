load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "mlx",
        build_file = Label("//third_party/mlx:mlx.bazel"),
        patch_args = ["-p0"],
        patches = [Label("//third_party/mlx:runtime_default_metallib.patch")],
        sha256 = "a4c83fc23ab6b376138be03c34a17ddb12aa2ee9f702343a7f9f25b96b12b614",
        strip_prefix = "mlx-7b7c12407f85b494e3e6d1cd3888650d224f362c",
        urls = [
            "https://github.com/ml-explore/mlx/archive/7b7c12407f85b494e3e6d1cd3888650d224f362c.zip",
        ],
    )
