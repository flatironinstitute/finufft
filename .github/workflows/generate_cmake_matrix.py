import json

matrix = {
    "include": []
}

build_type = ["Release", "Debug"]
static_linking = ["On", "Off"]
combinations = {
    "ubuntu-22.04": {
        "toolchain": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"]
    },
    "windows-2022": {
        "toolchain": ["msvc", "llvm"],
        "arch_flags": ["/arch:AVX2", "/arch:SSE2", "native"]
    },
    "macos-13": {
        "toolchain": ["llvm", "gcc-14"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"]
    }
}


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

for platform in combinations.keys():
    for toolchain in combinations[platform]["toolchain"]:
        for arch_flag in combinations[platform]["arch_flags"]:
            for linking in static_linking:
                for build in build_type:
                    matrix["include"].append({
                        "os": platform,
                        "toolchain": toolchain,
                        "arch_flags": arch_flag,
                        "finufft_static_linking": linking,
                        "build_type": build,
                        "c_compiler": get_c_compiler(toolchain),
                        "cxx_compiler": get_cxx_compiler(toolchain)
                    })
json_str = json.dumps(matrix, ensure_ascii=False)
print(json_str)
