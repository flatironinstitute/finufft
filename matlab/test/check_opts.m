% Checks that a mistyped opts field warns rather than being silently ignored,
% and that the fields the .m layer legitimately forwards stay silent. See #895.

M = 10; N = 8;
x = 2*pi*rand(M,1); c = randn(M,1) + 1i*randn(M,1);

lastwarn('');
o = struct('upsampfact', 2.0);              % typo for upsampfac
finufft1d1(x, c, +1, 1e-6, N, o);
[~, id] = lastwarn();
if ~strcmp(id, 'FINUFFT:unknownOpt')
  error('check_opts: mistyped opts field did not warn (id was "%s")', id);
end

lastwarn('');
o = struct('upsampfac', 2.0, 'debug', 0);   % floatprec is added by finufft1d1
finufft1d1(x, c, +1, 1e-6, N, o);
[~, id] = lastwarn();
if ~isempty(id)
  error('check_opts: valid opts warned unexpectedly ("%s")', id);
end

fprintf('check_opts: passed\n');
