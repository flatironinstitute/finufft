function Tx = applyToep(x,vhat)
% APPLYTOEP   fast matrix-vector multiply with square Toeplitz matrix
%
% Tx = applyToep(vhat,a) multiplies vector x by the square (generally
%  non-symmetric) Toeplitz matrix T defined by a vector v, whose DFT
%  vhat = fft(v) the user must supply. The convention for v (as in Raymond
%  Chan's book) is the 1st row of T in reverse order followed by the 2nd through
%  last elements of the 1st column in usual order. In the literature v is
%  indexed -N+1:N-1, where N is the matrix size. T*x is a discrete nonperiodic
%  convolution, and performed here by a FFT and iFFT pair.
%
% Inputs: x    : input column vector length N
%         vhat : DFT of v (length 2N-1)
% Output: Tx   : T*x, col vec length N
%
% Without arguments does self-test

% Barnett 11/7/22
if nargin==0, test_applyToep; return; end

N = numel(x);
assert(numel(vhat)==2*N-1)
xpadhat = fft(x(:),2*N-1);   % zero-pads out to size of vhat
Tx = ifft(xpadhat .* vhat(:));
Tx = Tx(N:end);              % extract correct chunk of padded output

%%%%%%%
function test_applyToep
N = 10;                  % size to compare against direct matvec
x = randn(N,1);
x = 0*x; x(1)=1;
t = randn(2*N-1,1);       % define nonsymm Toep: back 1st row then down 1st col
T = toeplitz(t(N:end),t(N:-1:1));   % munge single toep vec into (C,R) format
Tx = applyToep(x,fft(t));
fprintf('test_applyToep: Frob norm of diff btw fast and direct: %.3g\n',norm(T*x - Tx,'fro'))
