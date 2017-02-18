% plots kernel graphs from samples output from plotkernels.cpp
% Barnett 2/10/17
clear
a = textread('plotkernels.dat');
z = a(:,1); phi = a(:,2:end); [N nt] = size(phi);   % extract cols
maxphi = max(phi,[],1);
phin = phi./repmat(maxphi,[N 1]);    % peak normalize
figure(1); hold off; plot(z,phin,'.-');
figure(2); hold off; semilogy(z,phin,'.-');

% check asymptotics of I_0:
%z=100; besseli(0,z)/exp(z)*sqrt(2*pi*z)

stop
% compare against prev....
a = textread('plotkernels_kb.dat');
z = a(:,1); phi = a(:,2:end); [N nt] = size(phi);   % extract cols
maxphi = max(phi,[],1);
phin = phi./repmat(maxphi,[N 1]);    % peak normalize
figure(1); hold on; plot(z,phin,'-');
figure(2); hold on; semilogy(z,phin,'-');
