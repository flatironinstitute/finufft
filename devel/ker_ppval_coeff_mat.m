function C = ker_ppval_coeff_mat(w,d,be,o)
% KER_PPVAL_COEFF_MAT  matrix of piecewise poly ES kernel coeffs 
%
% C = ker_ppval_coeff_mat(w,nterms,beta,opts)
%
% Inputs:
%  w = integer kernel width in grid points, eg 10
%  d = poly degree to keep, eg 13
%  beta = kernel parameter, around 2.3*w
%  opts - optional struct [unused for now]
% Outputs:
%  C = (d+1)*w double-precision matrix of coeffs for ppval
%      Each col is (c_0...c_d) in c_0 + c_1z + ... + c_dz^d where |z|<=1
%      is the local variable in 1/w units about each grid pt.
%
% Notes: the self-test of this function is useful to tweak d (degree) for
%  each w.

% Barnett 4/23/18. Test all w, 7/16/24
if nargin==0, test_ker_ppval_coeff_mat; return; end
if nargin<4, o=[]; end

f = @(z) exp(be*(sqrt(1-z.^2)-1));    % ES kernel on [-1,1], handles complex

fitd = 20; % fit degree
m = 7;     % colloc pts per wall (4m>deg). Odd is better, hits Re axis
h = 1/w;   % size of half a grid spacing, in units where [-1,1] is kernel supp.
C = nan(fitd+1,w);  % alloc output
% set up collocation linear sys on a box...
l = 2*((1:m)'-0.5)/m-1;   % colloc pts on [-1,1], col vec
z = [l-1i; 1+1i*l; l+1i; -1+1i*l];  % colloc pts on complex bdry box [-1,1]^2
V = ones(4*m,fitd+1);
for k=1:fitd, V(:,k+1) = V(:,k) .* z; end   % fill Vandermonde
R = nan(4*m,w);     % stack all RHS in the lin sys...
for i=1:w
  xi = -1+h*(2*i-1);      % center of the i'th expansion, in [-1,1] supp.
  R(:,i) = f(xi+z*h);
end
C = V\R;            % do all solves for poly coeffs (multiple RHS)
C = C(1:d+1,:);     % keep only up to requested eval degree (coeffs 0 to d)

%%%%%%
function test_ker_ppval_coeff_mat
sigma = 2.0;  % upsampfac
for w=2:16
  if sigma==2.0          % we test the rules used in setup_spreader...
    betaoverws = [2.20 2.26 2.38 2.30];
    beta = betaoverws(min(4,w-1)) * w;
    tol = 10^(1-w);    % desired tol, in setup_spreader
    d = w+1 + (w<=8);  % hacking the degree rule so error << tol
  elseif sigma==1.25
    gamma = 0.97;
    betaoverns = gamma*pi*(1-1/(2*sigma));   % actually good for any sigma
    beta=betaoverns*w;
    tol = exp(-pi*w*sqrt(1-1/sigma));
    d = ceil(0.55*w+2.2);   % hacking the degree rule so error << tol
  end
  f = @(z) exp(beta*(sqrt(1-z.^2)-1));  % must match the above, to test

  C = ker_ppval_coeff_mat(w,d,beta);
  %C/exp(beta)  % shows that no advantage to truncating any tails...
  t = linspace(-1,1,30);   % list of offsets in 1/w units
  h = 1/w;
  maxerr = 0;   % track max error of evaluation in pts in all intervals
  for i=1:w
    xi = -1+h*(2*i-1);      % center of the i'th expansion, in [-1,1] supp.
    erri = max(abs(f(xi+t*h) - polyval(C(end:-1:1,i),t)));
    % erri = erri / exp(beta);     % scale to rel err to kernel peak (now 1)
    maxerr = max(erri,maxerr);
  end
  fprintf('w=%d\t d=%d\t tol=%.g\t maxerr=%.3g   \tmaxerr/tol=%.3g\n',w,d,tol,maxerr,maxerr/tol)
end
