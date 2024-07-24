import json

matrix = {
    "include": []
}

build_type = ["Release", "Debug"]
static_linking = ["On", "Off"]
combinations = {
    "ubuntu-22.04": {
        "compiler": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"]
    },
    "windows-2022": {
        "compiler": ["msvc", "llvm"],
        "arch_flags": ["/arch:AVX2", "/arch:SSE2", "native"]
    },
    "macos-13": {
        "compiler": ["llvm", "gcc-14"],
        "arch_flags": ["-march=native", "-march=x86-64", "native"]
    }
}

for platform in combinations.keys():
    for compiler in combinations[platform]["compiler"]:
        for arch_flag in combinations[platform]["arch_flags"]:
            for linking in static_linking:
                for build in build_type:
                    matrix["include"].append({
                        "os": platform,
                        "compiler": compiler,
                        "arch_flags": arch_flag,
                        "finufft_static_linking": linking,
                        "build_type": build
                    })
json_str = json.dumps(matrix, ensure_ascii=False)
print(json_str)
