#!/bin/bash
set -euo pipefail

rm -rf _build _stage _consume _leak

linking=${LINKING:-Static}
backend=${BACKEND:-ducc}
openmp=${OPENMP:-ON}
build_type=${BUILD_TYPE:-Release}
stage="$PWD/_stage"

static=ON
[[ "$linking" == "Shared" ]] && static=OFF
ducc=ON
[[ "$backend" == fftw* ]] && ducc=OFF

consumer=tools/ci/find_package-consumer
install_flags=(-DFINUFFT_USE_DUCC0=$ducc -DFINUFFT_STATIC_LINKING=$static
	-DFINUFFT_USE_OPENMP=$openmp)
# fftw-dl forces the downloaded FFTW: exercises the static-bundle install route.
[[ "$backend" == "fftw-dl" ]] && install_flags+=(-DFINUFFT_FFTW_LIBRARIES=DOWNLOAD)

cmake -S . -B _build -DCMAKE_BUILD_TYPE=$build_type \
	-DFINUFFT_ENABLE_INSTALL=ON \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _build --config $build_type
cmake --install _build --prefix "$stage" --config $build_type

run_app() { # $1 = the consumer's build directory
	local app
	for app in "$1/app" "$1/app.exe" "$1/Release/app.exe" "$1/Debug/app.exe"; do
		if [[ -x "$app" ]]; then
			"$app"
			return
		fi
	done
	echo "ERROR: $1 built no executable"
	exit 1
}

build_paths() { # $1 = install prefix, prints every offending line
	local targets
	targets=("$1"/lib*/cmake/finufft/finufftTargets*.cmake)
	[[ -f "${targets[0]}" ]] || {
		echo "ERROR: no exported targets file under $1, so the check proves nothing"
		exit 1
	}
	local patterns=(-e "$PWD" -e /usr/ -e /opt/ -e /home/ -e /Users/)
	if command -v cygpath >/dev/null; then
		patterns+=(-e "$(cygpath -m "$PWD")")
	fi
	grep -HF "${patterns[@]}" "${targets[@]}"
}
if build_paths "$stage"; then
	echo "ERROR: a build-machine path leaked into exported finufftTargets"
	exit 1
fi

cp -a "$stage" _leak
leak=(_leak/lib*/cmake/finufft/finufftTargets.cmake)
markers=("/usr/lib/libfftw3.so" "$PWD/libfinufft.a")
if command -v cygpath >/dev/null; then
	markers+=("$(cygpath -m "$PWD")/libfinufft.a")
fi
printf 'set(FINUFFT_LEAK_CONTROL "%s")\n' "${markers[@]}" >>"${leak[0]}"
hits=$(build_paths _leak) || {
	echo "ERROR: the build-machine path check does not fire on an injected leak"
	exit 1
}
for marker in "${markers[@]}"; do
	grep -qF "$marker" <<<"$hits" || {
		echo "ERROR: the build-machine path check does not fire on the injected $marker"
		exit 1
	}
done
rm -rf _leak

exported=OFF
grep -q "OpenMP::" "$stage"/lib*/cmake/finufft/finufftTargets.cmake && exported=ON
if [[ "$exported" != "$openmp" ]]; then
	echo "ERROR: built with FINUFFT_USE_OPENMP=$openmp, but the export says $exported"
	exit 1
fi

cmake -S "$consumer" -B _consume -DCMAKE_BUILD_TYPE=$build_type \
	-DCMAKE_PREFIX_PATH="$stage" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
cmake --build _consume --config $build_type

export LD_LIBRARY_PATH="$stage/lib:$stage/lib64:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$stage/lib:${DYLD_LIBRARY_PATH:-}"
run_app _consume
