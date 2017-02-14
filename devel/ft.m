function F = ft(f,L,k,q)
% compute real Fourier transform of even-symm func on [-L,L] at target freqs k,
% using direct summation with quadrature on [-L,L].
% q is optional override of even # quadr nodes.
kmax = max(abs(k));
if nargin<4
  q = ceil(20 + 1.5*kmax*L); if mod(q,2), q=q+1; end   % estimate q, even
end
[z w] = gauss(q); z = L*z(1:q/2); w = 2*L*w(1:q/2); % even symm quadr on [-L,L]
fj = f(z);      % func evals in one go
F = 0*k;        % same size as k list
for j=1:q/2
  F = F + w(j)*fj(j)*cos(k*z(j));    % real part since real and even symm
end
