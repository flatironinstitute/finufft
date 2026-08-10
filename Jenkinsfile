properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

// One entry per supported CUDA version. gcc is a host gcc-toolset this nvcc
// accepts. gpuType pins the card where the card is the point of the config, so
// a run always covers sm_80 and sm_120; the arch itself is read off the card
// rather than written down, and archs is what this nvcc can target - a pod that
// lands outside it skips instead of failing (sm_120 needs CUDA >= 12.8; 13.x
// dropped sm_70).
// One card of each kind, so no run duplicates an arch: 13.x cannot go on the
// V100 at all, which leaves 12.4 as the only entry that can still cover sm_70.
// The 11.8 entry is the unpinned one: it takes any GPU node and is the only one
// asking for two GPUs - four such pods would compete for whole nodes, and
// cufinufft_multigpu does not care which cards it gets (it SKIPs anyway unless
// they are full cards, since CUDA enumerates one MIG instance per process).
// Pinning that pod to the V100, the only pair of full cards, is what used to
// stall entire runs behind a busy card, three of them aborting at the timeout.
// Every minor here is one PyTorch ships wheels for, so the torch index is just
// the version without the dot; pick another and there will be no such index.
// Each pod needs a host driver its CUDA accepts (13.0 wants r580+); cuda-compat
// does not bridge a major version, so the first sh block prints nvidia-smi.
def configs = [
  [cuda: '11.8', gcc: 11,                     archs: ['70', '75', '80', '86', '89', '90'], gpus: 2],
  [cuda: '12.4', gcc: 13, gpuType: 'v100',    archs: ['70', '75', '80', '86', '89', '90']],
  [cuda: '12.8', gcc: 13, gpuType: 'rtx6000', archs: ['70', '75', '80', '86', '89', '90', '100', '120']],
  [cuda: '13.0', gcc: 14, gpuType: 'a100',    archs: ['75', '80', '86', '89', '90', '100', '120']],
]

catchError {
  timeout(time: 3, unit: 'HOURS') {
    buildImages(configs.collect { cfg ->
      [ context: 'tools/cufinufft/docker', dockerfile: 'Dockerfile-x86_64',
        tag: "cuda${cfg.cuda}",
        buildArgs: "--build-arg CUDA_VERSION=${cfg.cuda} --build-arg GCC_TOOLSET=${cfg.gcc}" +
                   " --build-arg TORCH_INDEX=cu${cfg.cuda.replace('.', '')}"
      ]
    }, checkout: true)

    parallel configs.collectEntries { cfg -> ['cuda-' + cfg.cuda, {
      runPod(tag: "cuda${cfg.cuda}", cpus: 8, memory: '32Gi',
             gpus: cfg.gpus ?: 1, gpuType: cfg.gpuType) {
        stage("cuda ${cfg.cuda}") {
          // compute_cap is "8.9" for sm_89 and "12.0" for sm_120.
          def arch = sh(returnStdout: true, script:
            'nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .').trim()
          if (!arch) {
            error "nvidia-smi did not report compute_cap - the whole matrix would skip silently"
          }
          if (!cfg.archs.contains(arch)) {
            echo "pod has sm_${arch}, which CUDA ${cfg.cuda} cannot target - skipping"
            return
          }
          withEnv([
            "HOME=$WORKSPACE",
            "CUDA=${cfg.cuda}",
            "CUDA_ARCH=${arch}",
            "LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64/stubs",
            "LD_LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64"
          ]) {
            sh '''
              nvidia-smi
              nvcc --version
              g++ --version
              cmake -B build . -DFINUFFT_USE_CUDA=ON \
                               -DFINUFFT_USE_CPU=OFF \
                               -DFINUFFT_BUILD_TESTS=ON \
                               -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
                               -DBUILD_TESTING=ON \
                               -DFINUFFT_STATIC_LINKING=OFF
              cmake --build build -j ${PARALLEL:-8}
              ctest --test-dir build/test/cuda --output-on-failure
            '''
            // Package with the release tooling (auditwheel, manylinux_2_28) and test
            // that wheel, not the build tree. Only the pod's own architecture is
            // compiled, unlike the released all-arch wheels.
            sh '''
              # The image already has auditwheel and the GPU frameworks; the venv is
              # only so the wheel installs somewhere writable.
              python3 -m venv --system-site-packages $HOME/venv
              source $HOME/venv/bin/activate
              tools/cufinufft/build-wheel.sh "$CUDA_ARCH" "wheelhouse/cuda$CUDA"
              python3 -m pip install --no-cache-dir wheelhouse/cuda$CUDA/cufinufft-*manylinux*.whl
              python3 -c "import cufinufft; print(cufinufft.__version__)"
            '''
            // catchError so a wheel that fails its tests still gets archived below.
            catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
              sh '''
                source $HOME/venv/bin/activate
                python3 -c "from numba import cuda; cuda.cudadrv.libs.test()"
                # From a copy outside the repo: pytest prepends the tests' rootdir to
                # sys.path, which would shadow the installed wheel with the source
                # tree. examples/ stays a sibling of tests/ (test_examples finds it
                # relative to itself).
                rm -rf $HOME/wheeltest && mkdir $HOME/wheeltest
                cp -r python/cufinufft/tests python/cufinufft/examples $HOME/wheeltest/
                cd $HOME/wheeltest
                # Every framework runs even if an earlier one fails, so the log says
                # which ones are broken rather than only the first.
                rc=0
                for framework in pycuda numba cupy torch; do
                  python3 -m pytest --framework=$framework tests || rc=$?
                done
                exit $rc
              '''
            }
            archiveArtifacts artifacts: 'wheelhouse/cuda*/cufinufft-*.whl', fingerprint: true
          }
        }
      }
    }] }
  }
}
emailFailure()
