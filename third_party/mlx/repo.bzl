load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "mlx_metal",
        build_file = Label("//third_party/mlx:mlx_metal.bazel"),
        sha256 = "84ffb60ee503f03eb684f5fb168d5cff31e2a16b7f27c1731eaf7662bd6e9b46",
        type = "zip",
        urls = [
            "https://files.pythonhosted.org/packages/99/82/11fd62a8d7a3e96e5c43220b17de0151e3f10101f8bb3b865f5bd9cdd074/mlx_metal-0.31.2-py3-none-macosx_26_0_arm64.whl",
        ],
    )
