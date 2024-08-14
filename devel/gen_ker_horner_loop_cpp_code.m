function str = gen_ker_horner_loop_cpp_code(w,d,be,o)
% GEN_KER_HORNER_LOOP_CPP_CODE  Write C++ for piecewise poly coeffs of kernel
%
% str = gen_ker_horner_loop_cpp_code(w,d,be,o)
%
% Inputs:
%  w = integer kernel width in grid points, eg 10
%  d = 0 for auto, else poly degree to keep, eg 13.  (will give # coeffs nc=d+1)
%  beta = kernel parameter, around 2.3*w (upsampfac=2 only)
%  opts - optional struct, with fields: [none for now].
%
% Outputs:
%  str = cell array of C++ code strings to define (d+1)*w static coeffs array,
%        as in each w-specific block in ../src/ker_horner_allw_loop_constexpr.h
%
% Also see: KER_PPVAL_COEFF_MAT, FIG_SPEED_KER_PPVAL (which tests acc too)
%           GEN_KER_HORNER_CPP_HEADER
%
% Notes:
%
% It exploits that there are w calls to different poly's w/ *same* z arg, writing
% this as a loop. This allows the kernel evals at one z to be a w*nc mat-vec.
% The xsimd code which uses this (see spreadinterp.cpp) is now too elaborate to
% explain here.
%
% Changes from gen_ker_horner_loop_C_code.m:
% i) a simple C++ style 2D array is written, with the w-direction fast, poly coeff
%    direction slow. (It used to be a list of 1D arrays plus Horner eval code.)
% ii) coeffs are now ordered c_d,c_{d-1},...,c_0, where degree d=nc-1. (Used
%    to be reversed.)
%
% Ideas: could try PSWF, cosh kernel ES variant, Reinecke tweaked-power ES
%  variant, etc..

% Ludvig af Klinteberg 4/25/18, based on Barnett 4/23/18. Ludvig wpad 1/31/20.
% Barnett redo for Barbone templated arrays, no wpad, 7/16/24.
% hardcoded outer nc array size, 7/22/24.

if nargin==0, test_gen_ker_horner_loop_cpp_code; return; end
if nargin<4, o=[]; end

C = ker_ppval_coeff_mat(w,d,be,o);
if d==0, d = size(C,2)-1; end
nc = d+1;
str = cell(nc+2,1);     % one coeff per line + one return and one close-paren line
% code to open the templated array...   why two {{?  (some C++ ambiguity thing)
str{1} = sprintf('  return std::array<std::array<T, w>, %d> {{\n', nc);
for n=1:d+1                  % loop over poly coeff powers 0,1,..,d
  % sprintf implicitly loops over fine-grid interpolation intervals 1:w...
  coeffrow = sprintf('%.16E, ', C(n,:));
  coeffrow = coeffrow(1:end-2);   % easy kill trailing comma (but allowed in C++)
  str{d+3-n} = sprintf('      {%s},\n', coeffrow);   % leave outer trailing comma
end
str{nc+2} = sprintf('  }};\n');     % terminate the array (two braces)


%%%%%%%%
function test_gen_ker_horner_loop_cpp_code  % writes code to file, doesn't test
w=13; d=w+1;           % pick a single kernel width and degree to write code for
%w=7; d=11;
%w=2; d=5;
beta=2.3*w;    % upsampfac=2 only
str = gen_ker_horner_loop_cpp_code(w,d,beta);
% str{:}
% check write and read to file...
fnam = sprintf('ker_horner_w%d.cpp',w);
fid = fopen(fnam,'w');
for i=1:numel(str); fwrite(fid,str{i}); end
fclose(fid);
system(['more ' fnam])
