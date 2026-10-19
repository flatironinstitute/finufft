properties([
  // abortPrevious, so a new push kills the superseded run instead of queueing
  // behind it: these builds are long and the old commit's result is not wanted.
  disableConcurrentBuilds(abortPrevious: true),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

// One entry per supported CUDA version. gcc is a host gcc-toolset this nvcc
// accepts. gpuType pins the card where the card is the point of the config, so
// a run always covers sm_80 and sm_120; the arch itself is read off the card
// rather than written down, and archs is what this nvcc can target - a pod that
// lands outside it skips instead of failing (sm_120 needs CUDA >= 12.8; 13.x
// dropped sm_70).
// The 11.8 entry is the unpinned one: it takes any GPU node and is the only one
// asking for two GPUs, since cufinufft_multigpu does not care which cards it
// gets (it SKIPs anyway unless they are full cards, as CUDA enumerates one MIG
// instance per process).
// torch names the PyTorch wheel index. It must match the toolkit, because the
// image puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which outranks torch's
// RUNPATH: a newer bundled cudart is shadowed by the older system one and every
// import dies on a missing symbol. Use 12.6 rather than 12.4, whose index
// stopped resolving when the cudnn its every torch pins was pruned from
// download.pytorch.org and pypi.nvidia.com alike.
// Each pod needs a host driver its CUDA accepts (13.0 wants r580+); cuda-compat
// does not bridge a major version, so the first sh block prints nvidia-smi.

// A perf pod takes the card rather than the node: only one node carries an
// rtx6000, so a request only that node could satisfy sits Pending until the
// timeout whenever anything else is on it.
//
// The rtx6000 and a100 nodes hand out MIG slices, and a slice shares its L2,
// its TLB and its link with the other slices, which is exactly what a timing
// must not do. The V100 is the highest capability this cluster hands over
// whole, so the perf pods take it and the preflight fails the stage if a device
// ever comes back named MIG.
//
// A whole-node pod is `exclusive: true`: a co-tenant pollutes a CPU measurement
// and pinning does not isolate it. The count has to stay next to it, because a
// cpu request still sets the pod's cpuset and runPod defaults to 4 - measured on
// build 50, whose exclusive pod got 4 of the node's 72 processors. The exclusive
// node has SMT off and 30 cores allocatable (SCC Jenkins wiki); the jnlp sidecar
// keeps 1.
// The harness sizes every case from the affinity mask it ends up with.
def PERF_CPU_CORES = 29

// gh is in the image; the credential is an environment variable of the step
// and never lives on disk. Publishing runs in the main container.
def withGh(Closure body) {
  container('main') {
    withCredentials([usernamePassword(credentialsId: 'github-jenkins',
                                      usernameVariable: 'GH_USER',
                                      passwordVariable: 'GH_TOKEN')]) {
      body()
    }
  }
}

// compute_cap is "8.9" for sm_89 and "12.0" for sm_120.
def gpuArch() {
  sh(returnStdout: true, script:
    'nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .').trim()
}

def configs = [
  [cuda: '11.8', gcc: 11, torch: 'cu118',                     archs: ['70', '75', '80', '86', '89', '90'], gpus: 2],
  [cuda: '12.6', gcc: 13, torch: 'cu126', gpuType: 'a100',    archs: ['70', '75', '80', '86', '89', '90']],
  [cuda: '12.8', gcc: 13, torch: 'cu128', gpuType: 'rtx6000', archs: ['70', '75', '80', '86', '89', '90', '100', '120']],
  [cuda: '13.0', gcc: 14, torch: 'cu130', gpuType: 'a100',    archs: ['75', '80', '86', '89', '90', '100', '120']],
]

catchError {
  timeout(time: 3, unit: 'HOURS') {
    // install cpu median 4980 s over 17 builds, serial make on 8 cpus.
    def installCpus = 8
    def jobs = configs.collectEntries { cfg -> ['cuda-' + cfg.cuda, {
      runPod(tag: "cuda${cfg.cuda}", cpus: installCpus, memory: '32Gi',
             gpus: cfg.gpus ?: 1, gpuType: cfg.gpuType) {
        stage("cuda ${cfg.cuda}") {
          def arch = gpuArch()
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
            "WITH_CUDA=ON",
            "LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64/stubs",
            "LD_LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64"
          ]) {
            sh 'tools/ci/cuda-build-test.sh'
            // The quick-start recipes against this checkout: install-test
            // builds and installs the tree, then consumes the staged install
            // with find_package (CPU and, behind WITH_CUDA, cuFINUFFT) and
            // FetchContent. The CPU arms alone also run on the install cpu
            // leg and on the host agents below.
            withEnv(["CMAKE_BUILD_PARALLEL_LEVEL=${installCpus}"]) {
              sh 'cmake -P tools/ci/install-test.cmake'
            }
            sh 'tools/ci/cuda-wheel.sh'
            // catchError so a wheel that fails its tests still gets archived below.
            catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
              sh 'tools/ci/cuda-wheel-test.sh'
            }
            archiveArtifacts artifacts: 'wheelhouse/cuda*/cufinufft-*.whl', fingerprint: true
          }
        }
      }
    }] }

    // The MEX against every MATLAB release MathWorks publishes an image for,
    // pulled as is: matlab: true injects the license, and the script fetches
    // the toolchain. matlab-deep-learning carries the Parallel Computing
    // Toolbox, so those releases also build and test the GPU MEX; r2020b has
    // only the plain image and runs the CPU half. gpuDevice() in r2024a fails
    // on a MIG slice (NVML Invalid Argument), so that release keeps the V100.
    // The license server has no r2026b yet.
    def matlabs = ['r2021b', 'r2022a', 'r2022b', 'r2023a', 'r2023b', 'r2024a',
                   'r2024b', 'r2025a', 'r2025b', 'r2026a']
    jobs['matlab-r2020b'] = {
      runPod(image: 'docker.io/mathworks/matlab:r2020b', cpus: 8, memory: '16Gi', matlab: true) {
        stage('matlab r2020b') {
          withEnv(["HOME=$WORKSPACE"]) { sh 'tools/ci/matlab-test.sh' }
        }
      }
    }
    // r2026a takes any GPU: it supports every compute capability. Its map
    // entry is null (runPod then asks for no card type); the rest take a100.
    def matlabGpuType = ['r2024a': 'v100', 'r2026a': null]
    matlabs.each { rel -> jobs['matlab-' + rel] = {
      runPod(image: "docker.io/mathworks/matlab-deep-learning:${rel}", cpus: 8, memory: '16Gi',
             gpus: 1,
             gpuType: matlabGpuType.containsKey(rel) ? matlabGpuType[rel] : 'a100',
             matlab: true) {
        stage("matlab ${rel}") {
          withEnv(["HOME=$WORKSPACE", "CUDA_ARCH=${gpuArch()}"]) { sh 'tools/ci/matlab-test.sh' }
        }
      }
    } }

    // The perftest comment, PR builds only: CHANGE_ID is unset on branch builds.
    // Each half writes its own section in its own pod - the CPU half wants cores
    // and no card, the GPU half wants the card - so the two run in parallel and
    // the post step concatenates the sections into one comment.
    //
    // Beside the validation matrix, not after it, so neither waits. A validation
    // pod may therefore land next to a perf pod; the interleaved rounds and the
    // per-arm median are what keep a noisy neighbour out of the ratio.
    //
    // UNSTABLE rather than failed, and catchInterruptions because the timeout is
    // the case to catch: a busy cluster must not mark the PR red.
    def perfCpu = {
      runPod(tag: 'cuda12.8', cpus: PERF_CPU_CORES, memory: '32Gi', exclusive: true) {
        stage('perf cpu') {
          withEnv([
            "HOME=$WORKSPACE",
            "CPM_SOURCE_CACHE=$WORKSPACE/.cpm"
          ]) {
            sh 'tools/ci/perf-cpu.sh'
          }
          stash name: 'perf-cpu', includes: 'cpu_perf.svg,cpu_perf.md'
        }
      }
    }

    // v100 for the card MIG cannot slice: the rtx6000 and a100 nodes run with MIG
    // on, and a slice shares its last-level TLB and PCIe with its neighbours.
    def perfGpu = {
      runPod(tag: 'cuda12.8', cpus: 32, memory: '32Gi', gpus: 1, gpuType: 'v100') {
        stage('perf gpu') {
          // Before the measuring, so a build that cannot post says so early.
          withGh { sh 'gh --version && gh auth status' }
          def arch = gpuArch()
          withEnv([
            "HOME=$WORKSPACE",
            "CUDA_ARCH=${arch}",
            "CPM_SOURCE_CACHE=$WORKSPACE/.cpm",
            "LIBRARY_PATH=/usr/local/cuda/lib64/stubs"
          ]) {
            sh 'tools/ci/perf-gpu.sh'
          }
          stash name: 'perf-gpu', includes: 'gpu_perf.svg,gpu_perf.md'
        }
      }
    }

    // Posting waits for both halves, so it is its own step, in a pod that needs
    // neither a card nor cores.
    def perfPost = {
      runPod(tag: 'cuda12.8', cpus: 2, memory: '8Gi') {
        stage('perf comment') {
          withGh {
            unstash 'perf-cpu'
            unstash 'perf-gpu'
            sh 'cat cpu_perf.md gpu_perf.md > perf_body.md'
            sh 'tools/ci/perf-plots-push.sh'
            sh 'gh pr comment "$CHANGE_ID" --repo flatironinstitute/finufft' +
               ' --edit-last --create-if-none --body-file perf_body.md'
            sh 'tools/ci/perf-refs-sweep.sh'
          }
        }
      }
    }

    def perfJob = {
      catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE',
                 catchInterruptions: true) {
        timeout(time: 75, unit: 'MINUTES') {
          parallel 'perf cpu': perfCpu, 'perf gpu': perfGpu
          perfPost()
        }
      }
    }

    // The perftest report page behind the RTD performance section. master
    // publishes it, and a PR has nothing to publish. A PR whose title carries
    // [perf page] rehearses the same job instead: everything runs and the
    // publish becomes an artifact, so a change to the page machinery is
    // provable on the PR rather than on the first master build after it lands.
    // Another branch of the same parallel, so it does not wait for the matrix.
    //
    // Two pods: one holds a whole node for both CPU backends in turn, one holds
    // the card. A post step assembles the page from the three stashes. UNSTABLE
    // rather than failed, since republishing a docs page says nothing about
    // library code.
    def pagePublishes = env.BRANCH_NAME == 'master'
    def PAGE_VERSIONS = 'v2.2.0 v2.3.1 v2.4.1 v2.5.1'

    // One backend, measured on a node this pod holds alone.
    def pageBackend = { backend, ducc ->
      stage('page ' + backend) {
        withEnv([
          "HOME=$WORKSPACE",
          "CPM_SOURCE_CACHE=$WORKSPACE/.cpm",
          "VERSIONS=${PAGE_VERSIONS}",
          "BACKEND=${backend}",
          "DUCC=${ducc}"
        ]) {
          sh 'tools/ci/page-worktrees.sh'
          sh 'tools/ci/page-cpu.sh'
        }
        stash name: 'page-' + backend, includes: 'outputs/**'
      }
    }

    // Both backends share one pod, one after the other. This cluster does not
    // hand out three whole nodes at once, so a pod per backend waits for a node
    // instead of running: slower, not faster. The pod still holds its node
    // alone, which is what the numbers need, and the two backends never appear
    // on one plot, so their order is free.
    def pageCpu = {
      runPod(tag: 'cuda12.8', cpus: PERF_CPU_CORES, memory: '32Gi', exclusive: true) {
        pageBackend('fftw', 'OFF')
        pageBackend('ducc', 'ON')
      }
    }

    // v100 for the card MIG cannot slice, as in the comment's GPU half. One
    // build per release, so the pod spends most of its time in nvcc.
    def pageGpu = {
      runPod(tag: 'cuda12.8', cpus: 32, memory: '32Gi', gpus: 1, gpuType: 'v100') {
        stage('page gpu') {
          def arch = gpuArch()
          withEnv([
            "HOME=$WORKSPACE",
            "CUDA_ARCH=${arch}",
            "CPM_SOURCE_CACHE=$WORKSPACE/.cpm",
            "LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
            "VERSIONS=${PAGE_VERSIONS}"
          ]) {
            sh 'tools/ci/page-worktrees.sh'
            sh 'tools/ci/page-gpu.sh'
          }
          stash name: 'page-gpu', includes: 'outputs/**'
        }
      }
    }

    def pagePost = {
      runPod(tag: 'cuda12.8', cpus: 2, memory: '8Gi') {
        stage('perftest page') {
          // Fail on the credential before the publish, not after.
          // A rehearsal publishes nothing, so it asks for no credential.
          if (pagePublishes) withGh { sh 'gh --version && gh auth status' }
          // A half that failed leaves its section out; the others still
          // publish. unstash of a missing stash is what would fail the step.
          for (half in ['page-fftw', 'page-ducc', 'page-gpu']) {
            catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
              unstash half
            }
          }
          sh 'tools/ci/page-assemble.sh'
          if (pagePublishes) withGh {
            // gh's credential helper writes the global gitconfig, and this
            // pod sets no home of its own: HOME is / there, and the build user
            // cannot write it.
            withEnv(["HOME=$WORKSPACE"]) {
              sh 'tools/ci/page-publish.sh'
            }
          } else {
            // A rehearsal stops here: the page and its figures ride out as
            // build artifacts, and the branch readthedocs fetches is untouched.
            archiveArtifacts artifacts: 'docs/performance.rst,docs/pics/perftestci_*.svg'
          }
        }
      }
    }

    def pageJob = {
      catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE',
                 catchInterruptions: true) {
        // Covers a wait for a free node, then both CPU backends in turn.
        timeout(time: 180, unit: 'MINUTES') {
          parallel 'page cpu': pageCpu,
                   'page gpu': pageGpu
          pagePost()
        }
      }
    }

    // Each half of either job holds a whole node or a whole card, so both at
    // once ask for two of each and the second whole-node pod stays
    // Unschedulable through its 300s launch timeout. In sequence instead.
    def measures = []
    if (env.CHANGE_ID) measures << perfJob
    if (pagePublishes || (env.CHANGE_TITLE ?: '').contains('[perf page]')) measures << pageJob
    if (measures) jobs['measure'] = { for (measure in measures) measure() }

    // The seven install arms run 32-33 min serially in one pod (Jenkins
    // jenkins-fix builds 4 and 6). Two pods split the arms between them. The
    // fftw-dl arm's _fetch consumer finds the system FFTW (DEFAULT), so the
    // arm downloads and builds FFTW once. A host without a system FFTW still
    // downloads it. See the FFTW note in tools/ci/install-test.cmake.
    def installLeg = { name, arms ->
      return {
        runPod(tag: 'cuda12.8', cpus: installCpus, memory: '16Gi') {
          stage(name) {
            withEnv(["CMAKE_BUILD_PARALLEL_LEVEL=${installCpus}"]) {
              for (arm in arms) {
                def linking = arm[0]
                def backend = arm[1]
                def openmp = arm[2]
                def armEnv = ["HOME=$WORKSPACE", "LINKING=${linking}", "BACKEND=${backend}",
                              "OPENMP=${openmp}"]
                if (arm.size() > 3) {
                  armEnv += ["BUILD_TYPE=${arm[3]}"]
                }
                withEnv(armEnv) {
                  sh 'cmake -P tools/ci/install-test.cmake'
                }
              }
            }
          }
        }
      }
    }
    // Measured arm times, Jenkins jenkins-fix build 9: leg 1 takes 3:58 +
    // 7:12 + 3:55 (about 15 min). Leg 2 takes 2:33 + 3:08 + 2:30 + 2:59
    // (about 11 min). The fftw-dl arm takes 7:12 in build 9 against 11:00 in
    // builds 4 and 6. The Debug tuple's fourth field sets BUILD_TYPE. That arm
    // covers the GCC -O1 workaround (xsimd 14.3.0 folds immediate arguments
    // wrongly at -O0, see cmake/toolchain.cmake).
    def installArms1 = [['Static', 'ducc', 'ON'], ['Static', 'fftw-dl', 'ON'],
                        ['Static', 'ducc', 'ON', 'Debug']]
    def installArms2 = [['Static', 'fftw', 'ON'], ['Shared', 'ducc', 'ON'],
                        ['Shared', 'fftw', 'ON'], ['Static', 'ducc', 'OFF']]
    ['install cpu 1': installArms1, 'install cpu 2': installArms2].each { n, arms ->
      jobs[n] = installLeg(n, arms)
    }

    // The conda recipe from docs/install.rst, run without change. /opt/conda
    // belongs to root, so conda keeps its environments and packages in the workspace.
    jobs['conda finufft'] = {
      runPod(image: 'docker.io/condaforge/miniforge3:latest', cpus: 8, memory: '16Gi') {
        stage('conda finufft') {
          withEnv(["HOME=$WORKSPACE",
                   "CONDA_ENVS_PATH=$WORKSPACE/.conda/envs",
                   "CONDA_PKGS_DIRS=$WORKSPACE/.conda/pkgs"]) {
            sh 'bash python/finufft/conda-install.sh'
          }
        }
      }
    }
    // CUDA 12.4 from conda cannot target sm_120, so the pod takes an a100.
    jobs['conda cufinufft'] = {
      runPod(image: 'docker.io/condaforge/miniforge3:latest', cpus: 8, memory: '32Gi',
             gpus: 1, gpuType: 'a100') {
        stage('conda cufinufft') {
          withEnv(["HOME=$WORKSPACE",
                   "CONDA_ENVS_PATH=$WORKSPACE/.conda/envs",
                   "CONDA_PKGS_DIRS=$WORKSPACE/.conda/pkgs"]) {
            sh 'bash python/cufinufft/conda-install.sh'
          }
        }
      }
    }

    // The quick-start recipes on the bare-metal host agents, CPU only: the
    // card lives in the k8s pods. install-test.cmake is the same install,
    // consume and run flow as on Linux; host-quick-start.* finds the tools on
    // the agent: the win script takes the VS-bundled cmake and loads
    // the MSVC environment from vcvars64.bat, the mac script downloads cmake
    // into CI_TOOLS, which sits in the build's own workspace. Jenkins gives
    // each concurrent build a fresh workspace, so two builds never share a
    // half-written cache. The macs' two arms share the download. OpenMP needs
    // libomp, which only the win agent's MSVC provides, so the macs run
    // single threaded.
    def quickStartHost = { label ->
      return {
        node(label) {
          stage("quick-start ${label}") {
            checkout scm
            def tools = isUnix() ? "${env.WORKSPACE}/.ci-tools" : "${env.WORKSPACE}\\.ci-tools"
            def arms = [['Static', 'OFF'], ['Shared', 'OFF']]
            if (label == 'win') {
              // The third arm builds static through clang-cl, the LLVM that
              // Visual Studio bundles beside MSVC.
              arms = [['Static', 'ON'], ['Shared', 'ON'], ['Static', 'ON', 'clang-cl']]
            }
            def bin = isUnix() ? "${tools}/bin" : "${tools}\\bin"
            withEnv(["CI_TOOLS=${tools}", "PATH+CI_TOOLS=${bin}"]) {
              for (arm in arms) {
                def armEnv = ["LINKING=${arm[0]}", "OPENMP=${arm[1]}", "BACKEND=ducc",
                              "WITH_CUDA=OFF"]
                if (arm.size() > 2) {
                  armEnv += ["CC=${arm[2]}", "CXX=${arm[2]}"]
                }
                withEnv(armEnv) {
                  if (isUnix()) {
                    sh 'tools/ci/host-quick-start.sh'
                  } else {
                    powershell 'tools/ci/host-quick-start.ps1'
                  }
                }
              }
            }
          }
        }
      }
    }
    def hostJobs = [:]
    for (label in ['win', 'macpro', 'macm1']) {
      hostJobs['quick-start ' + label] = quickStartHost(label)
    }

    // The host legs take bare agents and need no pod image, so they run beside
    // the image build instead of behind it. The pod legs keep their order:
    // buildImages first, then all their pods at once.
    parallel(
      'pod legs': {
        buildImages(configs.collect { cfg ->
          [ context: 'tools/cufinufft/docker', dockerfile: 'Dockerfile-x86_64',
            tag: "cuda${cfg.cuda}",
            buildArgs: "--build-arg CUDA_VERSION=${cfg.cuda} --build-arg GCC_TOOLSET=${cfg.gcc}" +
                       " --build-arg TORCH_INDEX=${cfg.torch}"
          ]
        }, checkout: true)
        parallel jobs
      },
      'host legs': {
        parallel hostJobs
      }
    )
  }
}
emailFailure()
