import json

matrix = {
    "include": []
}

python_versions = ["3.8", "3.11"]

combinations = {
    "ubuntu-22.04": {
        "compiler": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64"]
    },
    "windows-2022": {
        "compiler": ["msvc", "llvm"],
        "arch_flags": ["/arch:AVX2", "/arch:SSE2"]
    },
    "macos-13": {
        "compiler": ["llvm", "gcc-14"],
        "arch_flags": ["-march=native", "-march=x86-64"]
    }
}

for platform in combinations.keys():
    for python_version in python_versions:
        for compiler in combinations[platform]["compiler"]:
            for arch_flag in combinations[platform]["arch_flags"]:
                matrix["include"].append({
                    "os": platform,
                    "python-version": python_version,
                    "compiler": compiler,
                    "arch_flags": arch_flag
                })

json_str = json.dumps(matrix, ensure_ascii=False)
print(json_str)
