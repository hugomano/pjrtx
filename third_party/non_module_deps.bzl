load("//third_party/arocc:repo.bzl", arocc = "repo")
load("//third_party/fmt:repo.bzl", fmt = "repo")
load("//third_party/metal_cpp:repo.bzl", metal_cpp = "repo")
load("//third_party/mlx:repo.bzl", mlx = "repo")
load("//third_party/translate-c:repo.bzl", translate_c = "repo")
load("//third_party/xla:repo.bzl", xla = "repo")

def _non_module_deps_impl(mctx):
    xla()
    fmt()
    metal_cpp()
    mlx()
    arocc()
    translate_c()

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
