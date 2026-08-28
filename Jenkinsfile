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
// One card of each kind, so no run duplicates an arch: 13.x cannot go on the
// V100 at all, which leaves 12.6 as the only entry that can still cover sm_70.
// The 11.8 entry is the unpinned one: it takes any GPU node and is the only one
// asking for two GPUs, since cufinufft_multigpu does not care which cards it
// gets (it SKIPs anyway unless they are full cards, as CUDA enumerates one MIG
// instance per process).
// torch names the PyTorch wheel index. It must match the toolkit, because the
// image puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which outranks torch's
// RUNPATH: a newer bundled cudart is shadowed by the older system one and every
// import dies on a missing symbol. It also decides which architectures the wheel
// carries, and cu126 is the last index building for sm_70 - hence 12.6 for the
// V100 rather than 12.4, whose index stopped resolving when the cudnn its every
// torch pins was pruned from download.pytorch.org and pypi.nvidia.com alike.
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
// build 50, whose exclusive pod got 4 of the node's 72 processors. 64 is the
// most the current exclusive node (66 allocatable) can give while the jnlp
// sidecar keeps 1, and an even count means whole cores rather than SMT halves.
// The harness sizes every case from the affinity mask it ends up with.
def PERF_CPU_CORES = 64

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
  [cuda: '12.6', gcc: 13, torch: 'cu126', gpuType: 'v100',    archs: ['70', '75', '80', '86', '89', '90']],
  [cuda: '12.8', gcc: 13, torch: 'cu128', gpuType: 'rtx6000', archs: ['70', '75', '80', '86', '89', '90', '100', '120']],
  [cuda: '13.0', gcc: 14, torch: 'cu130', gpuType: 'a100',    archs: ['75', '80', '86', '89', '90', '100', '120']],
]

// The MATLAB release the Linux image installs. The Windows and macOS hosts run
// whatever SCC installed on them, and the stage prints it, so the three targets
// are not required to agree.
def MATLAB_RELEASE = 'R2025b'

// mpm installs MATLAB without a license, but starting MATLAB checks one out, so
// the license reaches the machine at run time and never enters an image. On the
// SCC hosts `module load matlab` sets MLM_LICENSE_FILE to the institute license
// manager; neither a Kubernetes pod nor an agent this pipeline provisioned has a
// module tree, so the value comes from a Jenkins credential.
//
// A credential rather than a global environment variable, for one reason: this
// console is world-readable without authentication, and Jenkins masks a
// credential in it. The address of the institute license manager is not
// something a public build log should carry.
//
// Secret text, id matlab-license, holding the <port>@<host> that
// `module load matlab` reports. Only the consuming half needs it: building the
// MEX never starts MATLAB.
def withMatlabLicense(Closure body) {
  withCredentials([string(credentialsId: 'matlab-license',
                          variable: 'MLM_LICENSE_FILE')]) {
    body()
  }
}

// The MATLAB image is 10 GB and takes a quarter of an hour to build, so the
// stage and its image are gated together: a topic branch neither builds nor
// pulls it.
//
// A branch named for this machinery is the exception: a change to the MATLAB
// stage has to be provable on the branch that makes it, without opening a pull
// request first.
def matlabRuns = env.CHANGE_ID || env.BRANCH_NAME == 'master' ||
                 (env.BRANCH_NAME ?: '').contains('matlab')

catchError {
  timeout(time: 3, unit: 'HOURS') {
    buildImages(configs.collect { cfg ->
      [ context: 'tools/cufinufft/docker', dockerfile: 'Dockerfile-x86_64',
        tag: "cuda${cfg.cuda}",
        buildArgs: "--build-arg CUDA_VERSION=${cfg.cuda} --build-arg GCC_TOOLSET=${cfg.gcc}" +
                   " --build-arg TORCH_INDEX=${cfg.torch}"
      ]
    } + (matlabRuns ? [
      [ context: 'tools/matlab/docker', dockerfile: 'Dockerfile-x86_64',
        tag: 'matlab',
        buildArgs: "--build-arg MATLAB_RELEASE=${MATLAB_RELEASE}"
      ],
      [ context: 'tools/matlab/docker', dockerfile: 'Dockerfile-consume',
        tag: 'matlab-consume',
        buildArgs: "--build-arg MATLAB_RELEASE=${MATLAB_RELEASE}"
      ]
    ] : []), checkout: true)

    def jobs = configs.collectEntries { cfg -> ['cuda-' + cfg.cuda, {
      runPod(tag: "cuda${cfg.cuda}", cpus: 8, memory: '32Gi',
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
            "LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64/stubs",
            "LD_LIBRARY_PATH=$WORKSPACE/build:/usr/local/cuda/lib64"
          ]) {
            sh 'tools/ci/cuda-build-test.sh'
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
            "CPM_SOURCE_CACHE=$WORKSPACE/.cpm",
            "PARALLEL=16"
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
            "LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
            "PARALLEL=12"
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
          "PARALLEL=16",
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
            "PARALLEL=12",
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
            archiveArtifacts artifacts: 'docs/performance_change_summary.rst,docs/pics/perftestci_*.svg'
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

    // Install and consume, the three routes a user takes: find_package against
    // an install, FetchContent against the sources, and a bare compiler line.
    // tools/ci/install-test.sh is the same script cmake_ci.yml runs on Windows
    // and the mac agents run below.
    //
    // One pod per half rather than one per cell: an install plus two consumers
    // is a couple of minutes, and four pods would spend longer being scheduled
    // than working. The arms run in sequence in a fresh directory each time.
    jobs['install cpu'] = {
      runPod(tag: 'cuda12.8', cpus: 8, memory: '16Gi') {
        stage('install cpu') {
          for (linking in ['Static', 'Shared']) {
            for (backend in ['ducc', 'fftw']) {
              // The FFTW controls need a configure each and no build, so they
              // ride on one arm rather than all four.
              def controls = (linking == 'Static' && backend == 'fftw') ? '1' : '0'
              withEnv(["HOME=$WORKSPACE", "LINKING=${linking}", "BACKEND=${backend}",
                       "CONTROLS=${controls}"]) {
                sh 'rm -rf _build _stage _consume _fetch _plain_app && tools/ci/install-test.sh'
              }
            }
          }
        }
      }
    }

    // The GPU twin, and the reason it is here rather than only on GitHub: a
    // runner has the CUDA toolkit but no device, so it can link the consumer and
    // never run it. This pod runs it.
    jobs['install cuda'] = {
      runPod(tag: 'cuda12.8', cpus: 8, memory: '16Gi', gpus: 1) {
        stage('install cuda') {
          def arch = gpuArch()
          if (!arch) {
            error "nvidia-smi did not report compute_cap - the consumer would build for the wrong card"
          }
          withEnv(["HOME=$WORKSPACE", "CUDA=1", "CUDA_ARCH=${arch}", "CONTROLS=1",
                   "LIBRARY_PATH=/usr/local/cuda/lib64/stubs"]) {
            sh 'tools/ci/install-test.sh'
          }
        }
      }
    }

    // MATLAB, in two pods, because the point is the route a user takes rather
    // than the build. The first has the toolchain, MATLAB and CUDA and produces
    // the CPack archive. The second is the mathworks/matlab image plus the
    // run-time libraries FINUFFT declares, with no compiler, no cmake and no
    // CUDA toolkit, and unpacks that archive and runs the interface tests
    // against it. A MEX that only works beside its own build tree fails there.
    //
    // Both take a card, so fullmathtest covers the GPU half. gpuType is unset on
    // purpose: correctness does not need a whole card, so these pods schedule on
    // any GPU node rather than queueing for the V100 the timing stages hold.
    if (matlabRuns) {
      jobs['matlab'] = {
        runPod(tag: 'matlab', cpus: 8, memory: '32Gi', gpus: 1) {
          stage('matlab build') {
            def arch = gpuArch()
            if (!arch) {
              error "nvidia-smi did not report compute_cap - the GPU half would skip silently"
            }
            withEnv([
              "HOME=$WORKSPACE",
              "CUDA_ARCH=${arch}",
              "FINUFFT_CI_GPU=1",
              "LIBRARY_PATH=/usr/local/cuda/lib64/stubs"
            ]) {
              sh 'tools/ci/matlab-build.sh'
            }
            stash name: 'matlab-mex', includes: 'finufft-matlab-mex-*.tar.gz'
            archiveArtifacts artifacts: 'finufft-matlab-mex-*.tar.gz', fingerprint: true
          }
        }
        runPod(tag: 'matlab-consume', cpus: 4, memory: '16Gi', gpus: 1) {
          stage('matlab consume') {
            unstash 'matlab-mex'
            withEnv(["HOME=$WORKSPACE", "FINUFFT_CI_GPU=1"]) {
              withMatlabLicense { sh 'tools/ci/matlab-consume.sh' }
            }
          }
        }
      }
    }

    // The install routes and the MATLAB interface on the three host agents.
    // Probed on 2026-08-28, all three carry a system compiler and nothing else:
    // no cmake, no ninja, no MATLAB, and win has no Visual Studio either.
    // agent-provision installs them, so the first build on an agent pays for the
    // ~10 GB MATLAB download and the rest find it already there.
    //
    // CI_TOOLS sits beside the workspace root rather than inside it: a workspace
    // belongs to one branch and is wiped, and an install that large must outlive
    // that. The two directories under it are what agent-provision produces:
    // bin holds cmake and ninja, matlab/<release> holds MATLAB.
    //
    // The macs run install-test.sh over the same four arms as the Linux pod, so
    // Intel and Apple silicon both cover find_package, FetchContent and the bare
    // compiler line. Windows does not, because install-test.sh needs the
    // compiler vcvars sets and only matlab-build.ps1 enters that shell;
    // cmake_ci.yml keeps the Windows consume until that is worth moving.
    //
    // CPU only. The hosts have no card, so matlab_test.m is left with
    // FINUFFT_CI_GPU unset and the pods above stay the only GPU coverage.
    //
    // Build then consume the MEX, as on Linux, but on one machine: there is no
    // second Windows or macOS agent to be the clean room, so these stages prove
    // the archive route without proving the absence of a toolchain.
    def matlabHost = { label ->
      return {
        node(label) {
          stage("host ${label}") {
            checkout scm
            // Cut the workspace root out of WORKSPACE rather than walking up
            // with '..': mpm rejects a --destination that is not canonical, and
            // build 8 spent all three agents finding that out.
            def cut = env.WORKSPACE.lastIndexOf('workspace')
            if (cut < 0) {
              error "WORKSPACE ${env.WORKSPACE} has no workspace component, so " +
                    "there is nowhere beside it to cache 10 GB of MATLAB"
            }
            def tools = env.WORKSPACE.substring(0, cut) + 'ci-tools'
            withEnv(["CI_TOOLS=${tools}",
                     "MATLAB_RELEASE=${MATLAB_RELEASE}",
                     "PATH+CI_TOOLS=${tools}/bin",
                     "PATH+MATLAB=${tools}/matlab/${MATLAB_RELEASE}/bin"]) {
              if (label == 'win') {
                powershell 'tools/ci/agent-provision.ps1'
                powershell 'tools/ci/matlab-build.ps1'
                withMatlabLicense { powershell 'tools/ci/matlab-consume.ps1' }
              } else {
                sh 'tools/ci/agent-provision.sh'
                for (linking in ['Static', 'Shared']) {
                  for (backend in ['ducc', 'fftw']) {
                    withEnv(["LINKING=${linking}", "BACKEND=${backend}"]) {
                      sh 'rm -rf _build _stage _consume _fetch _plain_app && tools/ci/install-test.sh'
                    }
                  }
                }
                sh 'tools/ci/matlab-build.sh'
                withMatlabLicense { sh 'tools/ci/matlab-consume.sh' }
              }
            }
          }
        }
      }
    }

    if (matlabRuns) {
      for (label in ['win', 'macpro', 'macm1']) {
        jobs['host ' + label] = matlabHost(label)
      }
    }

    parallel jobs
  }
}
emailFailure()
