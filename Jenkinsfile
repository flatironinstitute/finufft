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
            filename 'cufinufft/ci/docker/cuda11.0/Dockerfile-x86_64'
            args '--gpus 2'
         }
      }
      environment {
    HOME = "$WORKSPACE/build"
    PYBIN = "/opt/python/cp38-cp38/bin"
      }
      steps {
    sh '''#!/bin/bash -ex
      cp -r /io/build/test/cuda cuda_tests
      cd cuda_tests
      ctest --output-on-failure
    '''
    sh '${PYBIN}/python3 -m venv $HOME'
    sh '''#!/bin/bash -ex
      source $HOME/bin/activate
      python3 -m pip install --upgrade pip
      LIBRARY_PATH=/io/build python3 -m pip install -e cupython
      python3 -m pip install pytest
      python3 -m pytest
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
