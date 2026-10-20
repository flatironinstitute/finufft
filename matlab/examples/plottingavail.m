function isplot = plottingavail()
%PLOTTINGAVAIL True if plot display + graphics are available (MATLAB or Octave).
% so demo scripts do not crash on headless nodes
if exist('OCTAVE_VERSION', 'builtin') ~= 0
  isplot = ~isempty(getenv('DISPLAY'));
  isplot = isplot && ~isempty(available_graphics_toolkits());
else
  isplot = usejava('awt');   % MATLAB
end
