load("//third_party/arocc:repo.bzl", arocc = "repo")
load("//third_party/fmt:repo.bzl", fmt = "repo")
load("//third_party/jax:repo.bzl", jax_upstream = "repo")
load("//third_party/metal_cpp:repo.bzl", metal_cpp = "repo")
load("//third_party/mlx:repo.bzl", mlx_metal = "repo")
load("//third_party/translate-c:repo.bzl", translate_c = "repo")
load("//third_party/xla:repo.bzl", xla = "repo")

def _non_module_deps_impl(mctx):
    xla()
    jax_upstream()
    fmt()
    metal_cpp()
    mlx_metal()
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
