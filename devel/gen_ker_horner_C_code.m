function str = gen_ker_horner_C_code(w,d,be,o)
% GEN_KER_HORNER_C_CODE  Write C code strings for Horner eval of ES kernel
%
% str = gen_ker_horner_C_code(w,d,be,o)
%
% Inputs:
%  w = integer kernel width in grid points, eg 10
%  d = poly degree to keep, eg 13
%  beta = kernel parameter, around 2.3*w
%  opts - optional struct
%
% Outputs:
%  str = length-w cell array of C code strings to eval each segment of kernel
%
% Also see: KER_PPVAL_COEFF_MAT

% Barnett 4/23/18
if nargin==0, test_gen_ker_horner_C_code; return; end
if nargin<4, o=[]; end

C = ker_ppval_coeff_mat(w,d,be,o);
str = cell(w,1);
for i=1:w   % loop over segments of kernel
  s = sprintf('ker[%d] = ',i-1);
  for n=1:d
    s = [s sprintf('%.16E + z*(',C(n,i))];   % (n-1)th coeff for i'th segment
  end
  s = [s sprintf('%.16E)',C(end,i))];
  for n=1:d-1, s = [s sprintf(')')]; end  % close all parens
  s = [s sprintf(';\n')];          % terminate the C line, CR
  str{i} = s;
end

%%%%%%%%
function test_gen_ker_horner_C_code
w=13; d=16;
%w=7; d=11;
%w=2; d=5;
beta=2.3*w;
str = gen_ker_horner_C_code(w,d,beta);
% str{:}
fnam = sprintf('ker_horner_w%d.c',w);
fid = fopen(fnam,'w');
for i=1:numel(str); fwrite(fid,str{i}); end
fclose(fid);
system(['more ' fnam])

