N = 1;
M = 1;
plan = finufft_plan(2, [N N], 1, 1, 1e-9);

tic
for k = 1:1000
    x = 2*pi*rand(M,1);
    y = 2*pi*rand(M,1);
    c = randn(N)+1i*randn(N);
    plan.setpts(x,y);
    f = plan.execute(c);
end
toc

