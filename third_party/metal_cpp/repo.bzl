load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "metal_cpp",
        build_file = Label("//third_party/metal_cpp:metal_cpp.bazel"),
        sha256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4",
        strip_prefix = "metal-cpp",
        urls = [
            "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip",
        ],
    )
