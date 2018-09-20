% check FINUFFT matlab interface doesn't change current # omp threads.
% Barnett 9/20/18

clear     % choose params...
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-6;   % requested accuracy
o.fftw = 0;       % style of FFTW: 0 (ESTIMATE) vs 1 (MEASURE, slow but reuses)
o.upsampfac=1.25; % 2.0 (default) or 1.25 (low-RAM, small-FFT)
M       = 1e7;    % # of NU pts (in all dims)
N       = 1e6;    % # of modes (approx total, used in all dims)
x = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);

o.debug = 1;      % choose 1 for timing breakdown text output
matnthr = maxNumCompThreads(8);   % measure matlab's state
fprintf('maxNumCompThreads = %d\n',maxNumCompThreads)
fprintf('1D: opts.nthreads=0\n')
tic; [f ier] = finufft1d1(x,c,isign,eps,N,o); toc
o.nthreads=1; fprintf('1D: opts.nthreads=1\n')
tic; [f ier] = finufft1d1(x,c,isign,eps,N,o); toc
o.nthreads=0; fprintf('1D: opts.nthreads=0\n')
tic; [f ier] = finufft1d1(x,c,isign,eps,N,o); toc   % should use all avail
maxNumCompThreads(matnthr);  % restore matlab's state

% NOTE: the sad thing is that omp has an internal state (changed by
% omp_set_max_threads) that Matlab cannot see with maxNumCompThreads !
% However, maxNumCompThreads(n) does change this state.
