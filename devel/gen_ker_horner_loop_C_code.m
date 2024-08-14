function str = gen_ker_horner_loop_C_code(w,d,be,o)
% GEN_KER_HORNER_LOOP_C_CODE  Write C code for looped Horner eval of ES kernel
%
% str = gen_ker_horner_loop_C_code(w,d,be,o)
%
% Inputs:
%  w = integer kernel width in grid points, eg 10
%  d = 0 (auto), or poly degree to keep, eg 13. (passed to ker_ppval_coeff_mat)
%  beta = kernel parameter, around 2.3*w (for upsampfac=2)
%  opts - optional struct, with fields:
%         wpad - if true, pad the number of kernel eval (segments) to w=4n
%                for SIMD speed, esp. w/ GCC<=5.4
%         cutoff - desired coeff cutoff for this kernel, needed when d=0
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
% Barnett fixed bug where degree was d-1 not d throughout; auto-d opt, 7/22/24.
if nargin==0, test_gen_ker_horner_loop_C_code; return; end
if nargin<4, o=[]; end

C = ker_ppval_coeff_mat(w,d,be,o);
if d==0, d = size(C,2)-1; end
str = cell(d+2,1);
if isfield(o,'wpad') && o.wpad
  width = 4*ceil(w/4);
  C = [C zeros(size(C,1),width-w)];    % pad coeffs w/ 0, up to multiple of 4
else
  width = w;
end
for n=1:d+1                 % loop over poly coeff powers
  s = sprintf('constexpr FLT c%d[] = {%.16E',n-1, C(n,1));
  for i=2:width            % loop over segments
    s = sprintf('%s, %.16E', s, C(n,i));
  end
  str{n} = [s sprintf('};\n')];
end

s = sprintf('for (int i=0; i<%d; i++) ker[i] = ',width);
for n=1:d
  s = [s sprintf('c%d[i] + z*(',n-1)];   % (n-1)th coeff for i'th segment
end
s = [s sprintf('c%d[i]',d)];
for n=1:d, s = [s sprintf(')')]; end  % close all parens
s = [s sprintf(';\n')];          % terminate the C line, CR
str{d+2} = s;

%%%%%%%%
function test_gen_ker_horner_loop_C_code  % writes C code to file, doesn't test
w=13; d=0; opts.cutoff = 1e-12;   % pick a width and cutoff for degree
beta=2.3*w;        % implies upsampfac=2
%w=7; d=11;
%w=2; d=5;
str = gen_ker_horner_loop_C_code(w,d,beta,opts);
% str{:}
fnam = sprintf('ker_horner_w%d.c',w);
fid = fopen(fnam,'w');
for i=1:numel(str); fwrite(fid,str{i}); end
fclose(fid);
system(['more ' fnam])
