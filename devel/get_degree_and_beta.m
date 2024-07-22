function [d,beta] = get_degree_and_beta(w,upsampfac)
% GET_DEGREE_AND_BETA  defines degree & beta from w & upsampfac
%
% [d,beta] = get_degree_and_beta(w,upsampfac)
%
% Universal definition for piecewise poly degree chosen for kernel
% coeff generation by matlab, and the ES kernel beta parameter.
% The map from tol to width w must match code in spreadinterp used to
% choose w.
%
% Used by all other *.m codes for generating coeffs.
%
% To test: use KER_PPVAL_COEFF_MAT self-test
%
% To verify accuracy in practice, compile FINUFFT CPU then run
% test/checkallaccs.sh and matlab/test/fig_accuracy.m
%
% Also see: REVERSE_ENGINEER_TOL, KER_PPVAL_COEFF_MAT

% Barnett 7/22/24
if upsampfac==0.0, upsampfac=2.0; end

% if d set to 0 in following, means it gets auto-chosen...
if upsampfac==2    % hardwire the betas for this default case
  betaoverws = [2.20 2.26 2.38 2.30];   % must match setup_spreader
  beta = betaoverws(min(4,w-1)) * w;    % uses last entry for w>=5
  d = w + 1 + (w<=7) - (w==2);          % between 1-2 more degree than w. tweak
elseif upsampfac==1.25  % use formulae, must match params in setup_spreader
  gamma=0.97;                           % safety factor
  betaoverws = gamma*pi*(1-1/(2*upsampfac));  % from cutoff freq formula
  beta = betaoverws * w;
  d = ceil(0.7*w+1.3);                  % less, since beta smaller. tweak
  %d = 0;    % auto-choose override? No, too much jitter.
end

if d==0
  tol = reverse_engineer_tol(w,upsampfac);
  opts.cutoff = 0.5 * tol;    % fudge to get more poly-approx acc than tol
  C = ker_ppval_coeff_mat(w,0,beta,opts);    % do solve merely to get d
  d = size(C,1)-1;            % extract the auto-chosen d
end
