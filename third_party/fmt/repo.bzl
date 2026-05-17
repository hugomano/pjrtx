load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "fmt",
        build_file = Label("//third_party/fmt:fmt.bazel"),
        sha256 = "e6a2b43c6ec4779fb53b15b705f7c20b4eb9d5ae35784d610ac6daec30737019",
        strip_prefix = "fmt-12.1.0",
        urls = [
            "https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.zip",
        ],
    )
