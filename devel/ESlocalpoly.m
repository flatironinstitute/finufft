% Test Taylor rep of ES spreading func around centers spaced as fine grid.
% Barnett 4/18/18

clear
w = 3;       % width in fine pts
be = 2.3*w;   % kernel width param
f = @(z) exp(be*(sqrt(1-z.^2)-1));    % the ES kernel on [-1,1], for eval in C
%f = @(z) exp(-be)*cosh(be*(sqrt(1-z.^2)));
h = 2/w;             % fine grid spacing in [-1,1]
ms = 3:10;
er = nan(w,numel(ms));  % sup errors on intervals, converging in m
p = 20;   % poly fit degree
pe = 6;   % poly eval degree
mt = 50; t = 2*(0:mt)/mt-1;   % test pts on [-1,1], include end pts
%figure;
for i=1:w   % loop over interval numbers to test
  xi = -1 + h*(i-0.5);
  c = nan(p+1,numel(ms));  % coeffs
  for j=1:numel(ms), m=ms(j);
    l = gauss(m);
    l = 2*((1:m)'-0.5)/m-1;   % colloc pts on [-1,1], col vec
    z = [l-1i; 1+1i*l; l+1i; -1+1i*l];  % colloc pts on bdry box [-1,1]^2
    %z = cos(pi*(1:4*m)'/(2*m) + 1i); % Bernstein ellipse, but spills off [-1,1]
    %figure; plot(z,'.'); axis equal; stop
    V = ones(4*m,p+1);
    for k=2:p+1, V(:,k) = V(:,k-1) .* z; end
    c(:,j) = V\f(xi+z*h/2);              % solve
    % test sup norm of err of degree-pe poly over the real interval...
    er(i,j) = max(abs(f(xi+t*h/2) - polyval(c(pe+1:-1:1,j),t)));
  end
  %semilogy(ms,abs(c-c(:,end)),'+-'); xlabel('m'); ylabel('err in polt coeffs'); hold on; title(sprintf('i=%d',i)); drawnow
  %legend(num2cellstr(1:p)); v=axis;v(3:4)=[1e-16,1];axis(v);
end
figure; semilogy(ms,er,'+-');title(sprintf('w=%d,p=%d,pe=%d',w,p,pe))
hline(10^(-w+1))

% m=5 certainly converged for p=20
% m=7 for p=25

%p>=22: (m>=6 is converged)
%pe=5:  1e -3.5
%pe=10: 1e-8
%pe=15: 1e-13
%pe=20: 1e-15

%, so build look-up with p=22, via m=7. Or use Bernstein ellipse to match

% or prestore and give taylor for
% but is taylor the best expansion over [-1,1] at each degree?

% w=15: pe=16 to beat 1e-14
% w=13: pe=15 to beat 1e-12.
% w=10: pe=12
% w=7:  pe=10  be well below 1e-6
% w=4:  pe=7        "        1e-3
% w=3:  pe=6

% NFFT ppl could argue that now we can eval any spread func fast
% so it may as well be theirs.

% cosh(be.sqrt(1-z^2)) is slightly cleaner than exp(..)
% which is then seeming v close to sinh(..)
% which is close to sinh(..) / (..)  being the backwards K-B.
