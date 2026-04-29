pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  environment {
    IMAGE = "$REGISTRY_PREFIX/${JOB_NAME.toLowerCase()}:$BUILD_NUMBER"
    PARALLEL = 4
  }
  stages {
    stage('image') {
      agent {
        kubernetes {
          inheritFrom 'podman'
          defaultContainer 'main'
        }
      }
      steps {
        sh 'podman build -t $IMAGE . -f tools/cufinufft/docker/cuda11.2/Dockerfile-x86_64'
        sh 'podman push $IMAGE'
      }
    }
    stage('build') {
      agent {
        kubernetes {
          inheritFrom 'jnlp'
          yaml """
            spec:
              runtimeClassName: nvidia
              imagePullSecrets:
                - name: registry-auth
              nodeSelector:
                nvidia: v100
              containers:
                - name: main
                  image: $IMAGE
                  command: [sleep]
                  args: [99999]
                  securityContext:
                    runAsUser: 1000
                    runAsGroup: 1000
                  resources:
                    limits:
                      cpu: $PARALLEL
                      memory: 16Gi
                      nvidia.com/gpu: 2
          """
          defaultContainer 'main'
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
        make -j$PARALLEL
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
        python -c "import cufinufft"
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
      emailext subject: '$DEFAULT_SUBJECT',
           body: '$DEFAULT_CONTENT',
           recipientProviders: [
         [$class: 'DevelopersRecipientProvider'],
           ],
           replyTo: '$DEFAULT_REPLYTO',
           to: 'janden-vscholar@flatironinstitute.org'
    }
  }
}
