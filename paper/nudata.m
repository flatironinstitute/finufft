function [c x y z] = nudata(nudist,M)
% NUDATA.  make non-uniform data for NUFFT tests
%
% [c x y z] = nudata(nudist,M)
% [c x y] = nudata(nudist,M)
% [c x] = nudata(nudist,M)
%  outputs: c is some random strengths, and the NU locations are
%  in x (for 1D), x y (for 2D), x y z (for 3D).  The # output args chooses
%  the dimensionality.
%
% Without args, does 3d test and makes EPS figs

% Barnett 11/7/17
if nargin==0, test_nudata; return; end

dim = nargout-1;
c = randn(M,1) + 1i*randn(M,1);   % strengths
x = pi*(2*rand(M,1)-1);

if nudist==0  % quasi-uniform
  if dim>1, y = pi*(2*rand(M,1)-1); end
  if dim>2, z = pi*(2*rand(M,1)-1); end

elseif nudist==1  % radial (unif in 1D, 1/r density in 2D, 1/r^2 in 3D)
  if dim>1, r = pi*rand(M,1); phi = 2*pi*rand(M,1); end
  if dim==2
    x = r.*cos(phi);
    y = r.*sin(phi);
  elseif dim==3
    costh = 2*rand(M,1)-1; sinth = sqrt(1-costh.^2);
    x = r.*sinth.*cos(phi);
    y = r.*sinth.*sin(phi);
    z = r.*costh;
  end
  
else, error('nudist > 1 not yet implemented');
end

%%%%%%%%%%%
function test_nudata
for nudist=0:1
  [c x y z] = nudata(nudist,5e3);
  figure; plot3(x,y,z,'.');
  axis vis3d equal tight off; view(20,10);
  set(gcf,'paperposition',[0 0 4 4]);
  print(sprintf('nudist_%d.eps',nudist),'-depsc2')
end


