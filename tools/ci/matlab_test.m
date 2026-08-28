% Run the MATLAB interface tests, and fail if they tested nothing.
%
% Underscore, not the hyphen the sibling shell scripts use: MATLAB derives a
% script's name from its file name, and a hyphen is not a legal identifier, so
% run('tools/ci/matlab-test.m') would execute MATLAB's own matlab.m instead.
%
% fullmathtest only warns when it finds no MEX file, so an interface that failed
% to build leaves a green stage behind. This driver asserts up front what the
% target is expected to carry, then runs the tests.
%
% Environment:
%   FINUFFT_BUILD_DIR  directory holding the built MEX files (default 'build')
%   FINUFFT_CI_GPU     '1' to also require cufinufft and a usable device

% run() changes the current folder to the one holding the script, so a relative
% path here would resolve against tools/ci rather than the repo. This file's own
% location fixes the repo root instead.
cd(fullfile(fileparts(mfilename('fullpath')), '..', '..'));

buildDir = getenv('FINUFFT_BUILD_DIR');
if isempty(buildDir)
    buildDir = 'build';
end
wantGpu = strcmp(getenv('FINUFFT_CI_GPU'), '1');

% The build directory goes on first so addpath's prepend leaves the freshly
% built MEX ahead of any copy sitting in the source tree.
addpath(genpath('matlab'));
addpath(genpath(buildDir));

fprintf('MATLAB %s, matlabroot %s, mexext %s\n', version, matlabroot, mexext);
fprintf('computer %s, maxNumCompThreads %d\n', computer, maxNumCompThreads);

if exist('finufft') ~= 3
    error('finufft:ci:noCpuMex', ...
          'no CPU MEX named finufft.%s under %s', mexext, buildDir);
end
fprintf('CPU MEX: %s\n', which('finufft'));

if wantGpu
    if exist('cufinufft') ~= 3
        error('finufft:ci:noGpuMex', ...
              'FINUFFT_CI_GPU=1 but no GPU MEX named cufinufft.%s under %s', ...
              mexext, buildDir);
    end
    % canUseGPU needs the Parallel Computing Toolbox. Without a device the tests
    % would silently drop their GPU half, which is the failure this guards.
    if exist('canUseGPU') == 0 || ~canUseGPU()
        error('finufft:ci:noDevice', ...
              'FINUFFT_CI_GPU=1 but canUseGPU() is false: no device or no PCT');
    end
    d = gpuDevice();
    fprintf('GPU MEX: %s on %s (compute %s)\n', which('cufinufft'), d.Name, ...
            d.ComputeCapability);
end

fullmathtest
tolsweeptest
disp('matlab_test.m passed.');
