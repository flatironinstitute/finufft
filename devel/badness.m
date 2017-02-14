function b = badness(phi,L,R)
% err estimate (inverse figure of merit) for window func phi. small is good.
% phi should be even-symm and support on [-L,L]. R is the upsampling ratio.
nfreqs = 20;                % pretty much a guess, just dense enough
kuse = linspace(0,pi/R,nfreqs);   % central freq range
phihatuse = abs(ft(phi,L,kuse));             % phihat in central used range
phihatalias = 0*kuse;
nimg = 4;                      % how far to sum over nearby aliased copies
for m=-nimg:nimg, if m~=0      % some nearest aliased copies...
    phihatalias = phihatalias + abs(ft(phi,L,kuse+2*pi*m));
  end,end
%b = max(phihatalias./phihatuse);      % worst case over used freq range
b = norm(phihatalias./phihatuse)/sqrt(nfreqs);  % l2 over used freq range
