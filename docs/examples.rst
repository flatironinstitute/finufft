Examples and applications
=========================

Periodic Poisson solve on non-Cartesian grid
--------------------------------------------

It is well known that the FFT can be used to solve the Poisson
equation on a periodic cuboid domain, say $[0,2\pi)^d$, namely, given $f$,
to find $u$ satisfying

.. math::

   -\Delta u = f~, \qquad \mbox{ where } \int_{[0,2\pi)^d} f \, dx = 0~,

which has a unique solution up to constants. When $f$ and $u$ are on
a regular Cartesian mesh, three steps are needed.
The first takes an FFT to approximate
the Fourier series coefficent array of $f$, the second divides by $\|k\|^2$,
and the third uses another FFT to evaluate the Fourier series for $u$
back on the original grid. Here is a MATLAB demo in $d=2$ dimensions.
Firstly we set up a smooth function, periodic up to machine precision::

  w0 = 0.1;  % width of bumps
  src = @(x,y) exp(-0.5*((x-1).^2+(y-2).^2)/w0^2)-exp(-0.5*((x-3).^2+(y-5).^2)/w0^2);

Now we do the FFT solve, using a loop to check convergence with respect to
``n`` the number of grid points in each dimension::

  ns = 40:20:120;   % convergence study of grid points per side
  ns = 2*ceil(ns/2);      % insure even
  for i=1:numel(ns), n = ns(i);
    x = 2*pi*(0:n-1)/n;   % grid
    [xx yy] = ndgrid(x,x);   % ordering: x fast, y slow
    f = src(xx,yy);         % eval source on grid
    fhat = ifft2(f);        % Fourier series coeffs by Euler-F projection
    k = [0:n/2-1 -n/2:-1];   % Fourier mode grid
    [kx ky] = ndgrid(k,k);
    kfilter = 1./(kx.^2+ky.^2);    % inverse -Laplacian in Fourier space
    kfilter(1,1) = 0;   % kill the average, zero mode (even if inconsistent)
    kfilter(n/2+1,:) = 0; kfilter(:,n/2+1) = 0; % kill n/2 modes since non-symm
    u = fft2(kfilter.*fhat);   % do it
    u = real(u);
    fprintf('n=%d:\t\tu(0,0) = %.15e\n',n,u(1,1))   % check conv at a point
  end

We observe spectral convergence to 14 digits::

  n=40:		u(0,0) = 1.551906153625019e-03
  n=60:		u(0,0) = 1.549852227637310e-03
  n=80:		u(0,0) = 1.549852190998224e-03
  n=100:	u(0,0) = 1.549852191075839e-03
  n=120:	u(0,0) = 1.549852191075828e-03


