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
            filename 'tools/cufinufft/docker/cuda11.0/Dockerfile-x86_64'
            args '--gpus 2'
         }
      }
      environment {
    HOME = "$WORKSPACE"
    PYBIN = "/opt/python/cp38-cp38/bin"
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
        cuda_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1| sed "s/\\.//")
        cmake -B build . -DFINUFFT_USE_CUDA=ON \
                         -DFINUFFT_USE_CPU=OFF \
                         -DFINUFFT_BUILD_TESTS=ON \
                         -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
                         -DBUILD_TESTING=ON
        cd build
        make -j4
    '''
    sh '''#!/bin/bash -ex
      cd build/test/cuda
      ctest --output-on-failure
    '''
    sh '${PYBIN}/python3 -m venv $HOME'
    sh '''#!/bin/bash -ex
      source $HOME/bin/activate
      python3 -m pip install --upgrade pip
      python3 -m pip install -e python/cufinufft
      python3 -m pip install pytest
      python3 -m pytest python/cufinufft
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
