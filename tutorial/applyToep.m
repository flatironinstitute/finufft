function Tx = applyToep(x,vhat)
% APPLYTOEP   fast matrix-vector multiply with square Toeplitz matrix
%
% Tx = applyToep(x,vhat) multiplies vector x by the square N*N (generally
%  non-symmetric) Toeplitz matrix T defined by a vector v of length 2N-1
%  whose 2N-padded DFT vhat = fft([v;0]) the user must supply.
%  The convention for v (as in Raymond
%  Chan's book) is the 1st row of T in reverse order followed by the 2nd through
%  last elements of the 1st column in usual order. In the literature v is
%  indexed -N+1:N-1. T*x is a discrete nonperiodic
%  convolution, and performed here by a FFT and iFFT pair.
%  This version uses FFTs of size 2N instead of 2N-1, since the latter has much
%  larger factors (it is often prime) which slow down the FFT dramatically.
%
% Inputs: x    : input column vector length N
%         vhat : DFT of v after padding to length 2N (eg, by a single zero)
% Output: Tx   : T*x, col vec length N
%
% Without arguments does self-test; see this code for a demo of use

% Barnett 11/7/22. Realized 2N-1 slow for FFT (can be prime!) -> 2N.  12/10/23
if nargin==0, test_applyToep; return; end

N = numel(x);
assert(numel(vhat)==2*N)
xpadhat = fft(x(:),2*N);   % zero-pads out to size of vhat
Tx = ifft(xpadhat .* vhat(:));
Tx = Tx(N:end-1);              % extract correct chunk of padded output

%%%%%%%
function test_applyToep
N = 10;                   % size to compare against direct matvec
x = randn(N,1);
t = randn(2*N-1,1);       % define nonsymm Toep: back 1st row then down 1st col
T = toeplitz(t(N:end),t(N:-1:1));   % munge single toep vec into (C,R) format
tpad = [t;0]; that = fft(tpad);     % shows user how to pad
Tx = applyToep(x,that);
fprintf('test_applyToep: Frob norm of diff btw fast and direct: %.3g\n',norm(T*x - Tx,'fro'))
