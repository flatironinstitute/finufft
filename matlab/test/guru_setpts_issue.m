% Tests fix of setpts temporary array loss by MWrap (issue 185).
% The issue occurred when expressions such as -x were passed into setpts,
% resulting in crash or incorrect answers (due to pointing to destroyed temp
% arrays).
% It is fixed as of 5/6/2021.
% code by Dan Fortunato.

% Random points
M = 10000;
x = pi*(2*rand(M,1)-1);
y = pi*(2*rand(M,1)-1);

% Random Fourier coefficients
N = 64;
coeffs = randn(N) + 1i*randn(N);

% FINUFFT options
tol = 1e-12;
opts = struct();

for k = 1:100

    disp(k)

    % Without planning
    vals = finufft2d2(-x, -y, -1, tol, coeffs, opts);

    % With planning (was buggy, at seemingly random times)
    plan = finufft_plan(2, [N N], -1, 1, tol, opts);
    plan.setpts(-x, -y);
    vals2 = plan.execute(coeffs);

    % With planning (was the workaround, now not needed)
    plan = finufft_plan(2, [N N], -1, 1, tol, opts);
    xx = -x;
    yy = -y;
    plan.setpts(xx, yy);
    vals3 = plan.execute(coeffs);
    
    if ( any(isnan(vals2)) || norm(vals - vals2) > tol )
        warning('Something went wrong during run #%i', k);
        fprintf('norm(vals - vals2) = %g\n', norm(vals - vals2));
        fprintf('norm(vals - vals3) = %g\n', norm(vals - vals3));
        break
    end
end
