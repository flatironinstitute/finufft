import json

matrix = {
    "include": []
}

static_linking = ["On", "Off"]

combinations = [
    ("ubuntu-22.04", {
        "build_type": ["Release", "Debug"],
        "toolchain": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"],
        "ducc_fft": ["On", "Off"]
    }),
    ("windows-2022", {
        "build_type": ["Release", "Debug"],
        "toolchain": ["msvc"],
        "arch_flags": ["/arch:AVX2", "/arch:SSE2", "native"],
        "ducc_fft": ["On", "Off"]
    }),
    ("windows-2022", {
        "build_type": ["Release"],
        "toolchain": ["llvm"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"],
        "ducc_fft": ["On", "Off"]
    }),
    ("macos-13", {
        "build_type": ["Release", "Debug"],
        "toolchain": ["llvm", "gcc-14"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"],
        "ducc_fft": ["On", "Off"]
    })
]


def get_c_compiler(toolchain):
    if "gcc" in toolchain:
        return "gcc"
    elif toolchain == "llvm":
        return "clang"
    elif toolchain == "msvc":
        return "cl"
    else:
        raise ValueError(f"Unknown toolchain: {toolchain}")


def get_cxx_compiler(toolchain):
    if "gcc" in toolchain:
        return "g++"
    elif toolchain == "llvm":
        return "clang++"
    elif toolchain == "msvc":
        return "cl"
    else:
        raise ValueError(f"Unknown toolchain: {toolchain}")


for platform, value in combinations:
    for toolchain in value["toolchain"]:
        for arch_flag in value["arch_flags"]:
            for linking in static_linking:
                for build in value["build_type"]:
                    for ducc in value["ducc_fft"]:
                        matrix["include"].append({
                            "os": platform,
                            "toolchain": toolchain,
                            "arch_flags": arch_flag,
                            "finufft_static_linking": linking,
                            "build_type": build,
                            "c_compiler": get_c_compiler(toolchain),
                            "cxx_compiler": get_cxx_compiler(toolchain),
                            "ducc_fft": ducc
                        })
json_str = json.dumps(matrix, ensure_ascii=False)
print(json_str)
