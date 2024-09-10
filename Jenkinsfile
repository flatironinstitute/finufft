pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('main') {
      agent {
         dockerfile {
            filename 'tools/cufinufft/docker/cuda11.2/Dockerfile-x86_64'
            args '--gpus 2'
            label 'v100'
         }
      }
      environment {
    HOME = "$WORKSPACE"
    PYBIN = "/opt/python/cp310-cp310/bin"
    LIBRARY_PATH = "$WORKSPACE/build"
    LD_LIBRARY_PATH = "$WORKSPACE/build"
      }
      steps {
    sh '''#!/bin/bash -ex
      nvidia-smi
    '''
    sh '''#!/bin/bash -ex
      echo $HOME
    '''
    sh '''#!/bin/bash -ex
        # v100 cuda arch
        cuda_arch="70"

        cmake -B build . -DFINUFFT_USE_CUDA=ON \
                         -DFINUFFT_USE_CPU=OFF \
                         -DFINUFFT_BUILD_TESTS=ON \
                         -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
                         -DBUILD_TESTING=ON \
                         -DFINUFFT_STATIC_LINKING=OFF
        cd build
        make -j4
    '''
    sh '''#!/bin/bash -ex
      cd build/test/cuda
      ctest --output-on-failure
    '''
    sh '${PYBIN}/python3 -m venv $HOME'
    sh '''#!/bin/bash -ex
      cuda_arch="70"
      source $HOME/bin/activate

      python3 -m pip install --no-cache-dir --upgrade pip
      python3 -m pip install \
        --no-cache-dir \
        --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
        python/cufinufft
    '''
    sh '''#!/bin/bash -ex
      source $HOME/bin/activate
      python3 -m pip install --no-cache-dir --upgrade pycuda cupy-cuda112 numba
      python3 -m pip install --no-cache-dir torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
      python3 -m pip install --no-cache-dir pytest pytest-mock
      python -c "from numba import cuda; cuda.cudadrv.libs.test()"
      python3 -m pytest --framework=pycuda python/cufinufft
      python3 -m pytest --framework=numba python/cufinufft
      python3 -m pytest --framework=cupy python/cufinufft
      python3 -m pytest --framework=torch python/cufinufft
    '''
      }
    }
  }
  post {
    failure {
      emailext subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS',
           body: '''$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS

Check console output at $BUILD_URL to view full results.

Building $BRANCH_NAME for $CAUSE
$JOB_DESCRIPTION

Chages:
$CHANGES

End of build log:
${BUILD_LOG,maxLines=200}
''',
           recipientProviders: [
         [$class: 'DevelopersRecipientProvider'],
           ],
           replyTo: '$DEFAULT_REPLYTO',
           to: 'janden@flatironinstitute.org'
    }
  }
}
