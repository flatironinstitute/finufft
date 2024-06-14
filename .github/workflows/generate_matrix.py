import json

matrix = {
    "include": []
}

platforms = ["windows-latest", "macos-latest", "ubuntu-latest"]
python_versions = ["3.8", "3.11"]

combinations = {
    "ubuntu-latest": {
        "compiler": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64", ""]
    },
    "windows-latest": {
        "compiler": ["msvc"],
        "arch_flags": ["/arch:AVX2", "/arch:AVX512", "/arch:SSE2"]
    },
    "macos-latest": {
        "compiler": ["llvm", "gcc"],
        "arch_flags": ["-march=native", "-march=x86-64", ""]
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
