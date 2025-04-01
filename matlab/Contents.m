% FINUFFT: Flatiron Institute Nonuniform Fast Fourier Transform
% Version 2.4.0
%
% Basic and many-vector interfaces
%   finufft1d1 - 1D complex nonuniform FFT of type 1 (nonuniform to uniform).
%   finufft1d2 - 1D complex nonuniform FFT of type 2 (uniform to nonuniform).
%   finufft1d3 - 1D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%   finufft2d1 - 2D complex nonuniform FFT of type 1 (nonuniform to uniform).
%   finufft2d2 - 2D complex nonuniform FFT of type 2 (uniform to nonuniform).
%   finufft2d3 - 2D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%   finufft3d1 - 3D complex nonuniform FFT of type 1 (nonuniform to uniform).
%   finufft3d2 - 3D complex nonuniform FFT of type 2 (uniform to nonuniform).
%   finufft3d3 - 3D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%
% Guru interface (CPU)
%   finufft_plan - create guru plan object for one/many general nonuniform FFTs.
%   finufft_plan.setpts   - process nonuniform points for general transform(s).
%   finufft_plan.execute  - do single or many-vector transforms in a plan.
%
% If the GPU interface is installed (needs Parallel Computing Toolbox),
% the above nine basic and many-vector interfaces also handle gpuArrays,
% and the following GPU guru interface is available:
%   cufinufft_plan - create plan object for one/many general nonuniform FFTs.
%   cufinufft_plan.setpts  - input nonuniform points for general transform(s).
%   cufinufft_plan.execute - do single or many-vector transforms in a plan.
