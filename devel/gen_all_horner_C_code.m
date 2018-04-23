% script to make all C code for Horner eval of kernels of all widths.
% writes to "ker" array, from a variable "z", and switches by width "w".
% Resulting C code needs only placing in a function.
% Barnett 4/23/18

ws = 2:16;
fid = fopen('ker_horner_allw.c','w');
for j=1:numel(ws)
  w = ws(j)
  betaoverws = [2.20 2.26 2.38 2.30];   % matches setup_spreader
  beta = betaoverws(min(4,w-1)) * w;    % uses last entry for w>=5
  d = w + 2 + (w<8) + (w<12);    % between 2-4 more degree than w
  str = gen_ker_horner_C_code(w,d,beta);
  if j==1                                % write switch statement
    fwrite(fid,sprintf('  if (w==%d) {\n',w));
  else
    fwrite(fid,sprintf('  } else if (w==%d) {\n',w));
  end
  for i=1:numel(str); fwrite(fid,['    ',str{i}]); end
end
fwrite(fid,sprintf('  } else\n    printf("w not implemented!\\n");\n'));
fclose(fid);
