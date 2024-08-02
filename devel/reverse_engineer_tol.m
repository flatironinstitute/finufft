function tol = reverse_engineer_tol(w,upsampfac)
% REVERSE_ENGINEER_TOL  reconstructs tolerance from width and upsampfac
%
% tol = reverse_engineer_tol(w,upsampfac)
%
%  For fixed upsampfac (aka sigma), must be the inverse function for
%  how w is chosen from tol in spreadinterp.cpp:setup_spreader()
  
% Barnett 7/22/24
  
if upsampfac==2.0
  tol = 10^(1-w);
else
  tol = exp(-pi*w*sqrt(1-1/upsampfac));    % generic case, covers sigma=1.25
end
