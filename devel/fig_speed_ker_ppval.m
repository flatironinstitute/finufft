% script to generate and plot timing of raw kernel evals via various methods.
% Uses test_ker_ppval and its temp data file (see test_ker_ppval.cpp).
% Barnett 4/23/18

clear
nam = '/tmp/test_ker_ppval.dat';  % wipes any old data; make header for humans:
system(['echo "# M       w   t_plain  t_horner   relsuperr" > ' nam]);

Mwant=1e7;        % how many NU pts for a 1d1 or 1d2 NUFFT

ws=2:16;      % range of kernel widths, do timing tests...
for j=1:numel(ws), w=ws(j)
  % glib via shell must matter here, since links to glibc(?) w/o fast simd...
  % system(sprintf('./test_ker_ppval %d %d',Mwant,w));  % links to slower glibc?
  system(sprintf('(unset LD_LIBRARY_PATH; ./test_ker_ppval %d %d)',Mwant,w));
end

fid=fopen(nam,'r');   % read and make plot...
fgets(fid);      % ignore header line
[y,count] = fscanf(fid, '%f', [5,inf]);
fclose(fid);
if (count~=5*numel(ws)), warning('file wrong number of lines!'); end
y = y';              % since rows of text file come in as cols of array
M = y(:,1);
w = y(:,2);
r = (M.*w)./y(:,3);  % rate in evals/sec
r2 = (M.*w)./y(:,4); % "
e = y(:,5);          % rel err
figure; plot(w,[r r2]/1e6,'+-'); xlabel('w'); ylabel('eval rate (Meval/s)');
legend('exp eval','Horner'); title(sprintf('i7 GCC7.2 1thr ES kernel speed test, M=%d',Mwant))
print -dpng i7_1thr_ker_eval_speeds_loopversion.png


% xeon gcc6.4: exp max out at 40 Meval/s; horner 170-300 Meval/s.

% cf ludvig's i7 results: 0.2 sec for 1e7*(w=12) = 600 Meval/s
% (but that's special to m=12, also w/o the domain conditional?)
% Wouldn't it be nice if could get that for all i7 cases.

% Concl: for xeon w/ gcc, horner is much better! (5-10x)

