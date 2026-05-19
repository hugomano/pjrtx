load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "jax_upstream",
        build_file = "//third_party/jax:BUILD.jax_upstream.bazel",
        sha256 = "12ae17617d1346e2f98cfc48c1a000adc7389784eb119e8108a22dfd57cbb8c3",
        strip_prefix = "jax-jax-v0.10.0",
        urls = [
            "https://github.com/jax-ml/jax/archive/refs/tags/jax-v0.10.0.tar.gz",
        ],
    )
