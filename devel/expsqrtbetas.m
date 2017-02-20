% output all the best exp(sqrt) beta width params
% needs: gauss.m, badness.m, ft.m
% Barnett 2/14/17

clear
R = 2.0;   % overall upsampling fac
nss=2:15;
for i=1:numel(nss), ns=nss(i);
  fes = @(beta,x) exp(beta*sqrt(1-(2*x/ns).^2))/exp(beta)./sqrt(sqrt(1-(2*x/ns).^2));
  [betas(i) bs(i)] = fminbnd(@(beta) badness(@(x) fes(beta,x),ns/2,R),1.9*ns,2.4*ns);
  fprintf('%d\t%.6g\t%.6g \t beta/ns=%.6g\n',ns,betas(i),bs(i),betas(i)/ns);
end

figure; semilogy(nss,bs,'+'); xlabel('ns'); ylabel('est err (badness)');

f = fopen('expsqrtbetas.txt','w');
fprintf(f,'// for ns = %d to %d :\n\n',min(nss),max(nss))
fprintf(f,'const double betaoverns[] = {');  
for i=1:numel(nss), fprintf(f,'%.4g, ',betas(i)/nss(i)); end
fprintf(f,'};\n\n');
fprintf(f,'const double esterrs[] = {');
for i=1:numel(nss), fprintf(f,'%.3g, ',bs(i)); end
fprintf(f,'};');
fclose(f);
