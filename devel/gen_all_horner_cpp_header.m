% Script to make C++ header for looped Horner eval of kernels of all widths,
% for a particular opts.upsampfac (user to hand-choose below).
% C++ version (now writes .h), uses constexpr to switch by width w.

% Barnett 4/23/18; now calling Ludvig's loop version from 4/25/18.
% version including low upsampfac, 6/17/18.
% Ludvig put in w=4n padding, 1/31/20. Mystery about why d was bigger 2/6/20.
% C++ header using constexpr of Barbone, replacing *_C_code.m. Barnett 7/16/24.
clear
opts = struct();

ws = 2:16;           % list of widths (the driver, rather than toloerances)
upsampfac = 2.0;       % sigma (upsampling): either 2 (default) or low (eg 5/4).

if upsampfac==2
  fid = fopen('../src/ker_horner_allw_loop_constexpr.h','w');
  get_nc_code = 'w + 2 + (w <= 8)';   % must be C++ expression for cn=d+1
elseif upsampfac==1.25
  fid = fopen('../src/ker_lowupsampfac_horner_allw_loop_constexpr.h','w');
  get_nc_code = 'std::ceil(0.55*w + 3.2)';  % must be C++ code for nc=d+1
end
fwrite(fid,sprintf('// Header of static arrays of monomial coeffs of spreading kernel function in each\n'));
fwrite(fid,sprintf('// fine-grid interval. Generated by gen_all_horner_cpp_header.m in finufft/devel\n'));
fwrite(fid,sprintf('// Authors: Alex Barnett, Ludvig af Klinteberg, Marco Barbone & Libin Lu.\n// (C) 2018--2024 The Simons Foundation, Inc.\n'));
fwrite(fid,sprintf('#include <array>\n\n'));

usf_tag = sprintf('%d',100*upsampfac);  % follow Barbone convention: 200 or 125
fwrite(fid,sprintf('template<uint8_t w> constexpr auto nc%s() noexcept { return %s; }\n\n', usf_tag, get_nc_code));
fwrite(fid,sprintf('template<class T, uint8_t w>\nconstexpr std::array<std::array<T, w>, nc%s<w>()> get_horner_coeffs_%s() noexcept {\n',usf_tag,usf_tag));
fwrite(fid,sprintf('    constexpr auto nc = nc%s<w>();\n',usf_tag));

for j=1:numel(ws)
  w = ws(j)
  if upsampfac==2    % hardwire the betas for this default case
    betaoverws = [2.20 2.26 2.38 2.30];   % must match setup_spreader
    beta = betaoverws(min(4,w-1)) * w;    % uses last entry for w>=5
    d = w + 1 + (w<=8);                   % between 2-3 more degree than w
  elseif upsampfac==1.25  % use formulae, must match params in setup_spreader
    gamma=0.97;                           % safety factor
    betaoverws = gamma*pi*(1-1/(2*upsampfac));  % from cutoff freq formula
    beta = betaoverws * w;
    d = ceil(0.55*w+2.2);                  % less, since beta smaller, smoother
  end
  
  str = gen_ker_horner_loop_cpp_code(w,d,beta,opts);  % code strings for this w

  if j==1                                % write switch statement
    fwrite(fid,sprintf('    if constexpr (w==%d) {\n',w));
  else
    fwrite(fid,sprintf('    } else if (w==%d) {\n',w));
  end
  for i=1:numel(str); fwrite(fid,['    ',str{i}]); end   % format 4 extra spaces
end

% handle bad w at compile time...
fwrite(fid,sprintf('    } else {\n'));
fwrite(fid,sprintf('      static_assert(w >= %d, "w must be >= %d");\n',ws(1),ws(1)));
fwrite(fid,sprintf('      static_assert(w <= %d, "w must be <= %d");\n',ws(end),ws(end)));
fwrite(fid,sprintf('      return {};\n    }\n};\n'));    % close all brackets
fclose(fid);
