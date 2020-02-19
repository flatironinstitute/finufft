function str = gen_ker_horner_loop_C_code(w,d,be,o)
% GEN_KER_HORNER_LOOP_C_CODE  Write C code for looped Horner eval of ES kernel
%
% str = gen_ker_horner_loop_C_code(w,d,be,o)
%
% Inputs:
%  w = integer kernel width in grid points, eg 10
%  d = poly degree to keep, eg 13
%  beta = kernel parameter, around 2.3*w
%  opts - optional struct, with fields:
%         wpad - if true, pad the number of kernel eval (segments) to w=4n
%                for SIMD speed, esp. w/ GCC<=5.4
%         [ideas: could use to switch to cosh kernel variant, etc..]
%
% Outputs:
%  str = length-w cell array of C code strings to eval each segment of kernel
%
% Also see: KER_PPVAL_COEFF_MAT, FIG_SPEED_KER_PPVAL (which tests acc too)
%           GEN_KER_HORNER_C_CODE
%
% Exploits that there are w calls to different poly's w/ *same* z arg, writing
% this as a loop. This allows the compiler to vectorize in the w direction.
% (Horner can't be vectorized in the degree direction; Estrin was no faster.)

% Ludvig af Klinteberg 4/25/18, based on Barnett 4/23/18. Ludvig wpad 1/31/20.
if nargin==0, test_gen_ker_horner_loop_C_code; return; end
if nargin<4, o=[]; end

C = ker_ppval_coeff_mat(w,d,be,o);
str = cell(d+1,1);
if isfield(o,'wpad') && o.wpad
  width = 4*ceil(w/4);
  C = [C zeros(size(C,1),width-w)];    % pad coeffs w/ 0, up to multiple of 4
else
  width = w;
end
for n=1:d                  % loop over poly coeff powers
  s = sprintf('FLT c%d[] = {%.16E',n-1, C(n,1));
  for i=2:width            % loop over segments
    s = sprintf('%s, %.16E', s, C(n,i));      
  end
  str{n} = [s sprintf('};\n')];
end

s = sprintf('for (int i=0; i<%d; i++) ker[i] = ',width);
for n=1:d-1
  s = [s sprintf('c%d[i] + z*(',n-1)];   % (n-1)th coeff for i'th segment
end
s = [s sprintf('c%d[i]',d-1)];
for n=1:d-1, s = [s sprintf(')')]; end  % close all parens
s = [s sprintf(';\n')];          % terminate the C line, CR
str{d+1} = s;

%%%%%%%%
function test_gen_ker_horner_loop_C_code  % writes C code to file, doesn't test
w=13; d=16;           % pick a single kernel width and degree to write code for
%w=7; d=11;
%w=2; d=5;
beta=2.3*w;
str = gen_ker_horner_loop_C_code(w,d,beta);
% str{:}
fnam = sprintf('ker_horner_w%d.c',w);
fid = fopen(fnam,'w');
for i=1:numel(str); fwrite(fid,str{i}); end
fclose(fid);
system(['more ' fnam])
