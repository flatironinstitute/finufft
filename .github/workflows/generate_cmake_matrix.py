import json

matrix = {"include": []}

combinations = [
    (
        "ubuntu-22.04",
        [
            "gcc-10",
            "gcc-11",
            "gcc-12",
            "gcc-13",
            "clang-16",
            "clang-17",
            "clang-18",
        ],
    ),
    (
        "windows-2022",
        ["msvc", "clang-19"],
    ),
    (
        "macos-14",
        ["llvm", "gcc-14"],
    ),
]

def get_c_compiler(toolchain: str) -> str:
    if toolchain.startswith("gcc"):
        return "gcc"
    if toolchain == "llvm" or toolchain.startswith("clang"):
        return "clang"
    if toolchain.startswith("msvc"):
        return "cl"
    raise ValueError(f"Unknown toolchain: {toolchain}")


def get_cxx_compiler(toolchain: str) -> str:
    if toolchain.startswith("gcc"):
        return "g++"
    if toolchain == "llvm" or toolchain.startswith("clang"):
        return "clang++"
    if toolchain.startswith("msvc"):
        return "cl"
    raise ValueError(f"Unknown toolchain: {toolchain}")

for platform, toolchains in combinations:
    for toolchain in toolchains:
        matrix["include"].append(
            {
                "os": platform,
                "toolchain": toolchain,
                "c_compiler": get_c_compiler(toolchain),
                "cxx_compiler": get_cxx_compiler(toolchain),
            }
        )

print(json.dumps(matrix, ensure_ascii=False))
