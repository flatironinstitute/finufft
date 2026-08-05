properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

// One entry per supported CUDA version. gcc is a host gcc-toolset this nvcc
// accepts; arch must match gpuType (sm_120 needs CUDA >= 12.8, and CUDA 13
// dropped sm_70, so 13.x cannot run on the V100 pod). gpus: 2 only where full
// cards exist - CUDA enumerates a single MIG instance per process, so
// cufinufft_multigpu only compares across devices on the V100 pod.
// Every minor here is one PyTorch ships wheels for, so the torch index is just
// the version without the dot; pick another and there will be no such index.
// Each pod needs a host driver its CUDA accepts (13.0 wants r580+); cuda-compat
// does not bridge a major version, so the first sh block prints nvidia-smi.
def configs = [
  [cuda: '11.8', gcc: 11, gpuType: 'v100',    arch: '70',  gpus: 2],
  [cuda: '12.4', gcc: 13, gpuType: 'a100',    arch: '80',  gpus: 1],
  [cuda: '12.8', gcc: 13, gpuType: 'rtx6000', arch: '120', gpus: 1],
  [cuda: '13.0', gcc: 14, gpuType: 'a100',    arch: '80',  gpus: 1],
]

def testCuda(cfg) {
  buildPod(dockerfile: 'tools/cufinufft/docker/Dockerfile-x86_64',
           tag: "cuda${cfg.cuda}",
           buildArgs: "--build-arg CUDA_VERSION=${cfg.cuda} --build-arg GCC_TOOLSET=${cfg.gcc}" +
                      " --build-arg TORCH_INDEX=cu${cfg.cuda.replace('.', '')}",
           cpus: 8, memory: '32Gi', gpus: cfg.gpus, gpuType: cfg.gpuType) {
    stage("cuda ${cfg.cuda}") {
      withEnv([
        "HOME=$WORKSPACE",
        "CUDA=${cfg.cuda}",
        "CUDA_ARCH=${cfg.arch}",
        "LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64/stubs",
        "LD_LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64"
      ]) {
        sh '''#!/bin/bash -ex
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
        sh '''#!/bin/bash -ex
          # The image already has auditwheel and the GPU frameworks; the venv is
          # only so the wheel installs somewhere writable.
          python3 -m venv --system-site-packages $HOME/venv
          source $HOME/venv/bin/activate
          tools/cufinufft/build-wheel.sh "$CUDA_ARCH" "wheelhouse/cuda$CUDA"
          python3 -m pip install --no-cache-dir wheelhouse/cuda$CUDA/cufinufft-*manylinux*.whl
          python3 -c "import cufinufft; print(cufinufft.__version__)"
        '''
        // Before the tests: a wheel that fails them is the one worth keeping.
        archiveArtifacts artifacts: 'wheelhouse/cuda*/cufinufft-*.whl', fingerprint: true
        sh '''#!/bin/bash -ex
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
    }
  }
}

try {
  timeout(time: 3, unit: 'HOURS') {
    parallel configs.collectEntries { cfg -> ['cuda-' + cfg.cuda, { testCuda(cfg) }] }
  }
}
finally {
  emailFailure()
}
