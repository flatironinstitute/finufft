% FINUFFT: Flatiron Institute Nonuniform Fast Fourier Transform
% Version 1.2.0 11-Jun-2020
%
% Basic and many-vector interfaces
%   finufft1d1      - 1D type 1 complex nonuniform FFT (nonuniform to uniform)
%   finufft1d2      - 1D transform(s) of type 2 (uniform to nonuniform)
%   finufft1d3      - 1D transform(s) of type 3 (nonuniform to nonuniform)
%   finufft2d1      - 2D transform(s) of type 1 (nonuniform to uniform)
%   finufft2d2      - 2D transform(s) of type 2 (uniform to nonuniform)
%   finufft2d3      - 2D transform(s) of type 3 (nonuniform to nonuniform)
%   finufft3d1      - 3D transform(s) of type 1 (nonuniform to uniform)
%   finufft3d2      - 3D transform(s) of type 2 (uniform to nonuniform)
%   finufft3d3      - 3D transform(s) of type 3 (nonuniform to nonuniform)
%
% Guru interface
%   finufft_plan    - generate a plan object to perform general transform(s)
%   finufft_setpts  - process nonuniform points for general transform(s)
%   finufft_exec    - execute single or many-vector general transforms in a plan
%   finufft_destroy - deallocate (delete) a plan
