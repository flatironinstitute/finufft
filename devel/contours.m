% explore contours for saddle proof

clear
r=0.97; t0=r/sqrt(1-r^2);
a=(1-t0^2)/2; R = (1+t0^2)/2;
n = 1e3; t = (0.5:n-0.5)/n*2*pi;
z2 = a+R*exp(1i*t);
z = 1i*sqrt(-z2);   % branch cut at angle 0
figure;
plot(real(z),imag(z),'-'); hold on; zends=[z(1) z(end)];
plot(real(zends),imag(zends),'.','markersize',10); axis equal
title(sprintf('\\rho=%g\n',r))
