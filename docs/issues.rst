Known Issues
============

One should also check the github issues for the project page,
https://github.com/flatironinstitute/finufft/issues

Also see notes in the ``TODO`` file.

Issues with library
*******************

- When requested accuracy is ``1e-14`` or less, it is sometimes not possible to match this, especially when there are a large number of input and/or output points. This is believed to be unavoidable round-off error.

- Currently in Mac OSX, ``make lib`` fails to make the shared object library (.so).

- The timing of the first FFTW call is complicated, depending on whether FFTW_ESTIMATE (the default) or FFTW_MEASURE is used. Such issues are known, and discussed in other documentation, eg https://pythonhosted.org/poppy/fft_optimization.html .
  We would like to find a way of pre-storing some Intel-specific FFTW plans (as MATLAB does) to avoid the large FFTW planning times.
  
- Currently, a single library name is used for single- and multi-threaded versions. Thus, i) you need to ``make clean`` before changing such make options, and ii) if you wish to maintain multiple such versions you need to move them around and maintain them yourself, eg by duplicating the directory.

- The overhead for small problem sizes (<10000 data points) is too high, due to things such as the delay in FFTW looking up pre-stored wisdom. A unified advanced interface with a plan stage is in the works.
    
  
Issues with interfaces
**********************

- MATLAB, octave and python cannot exceed input or output data sizes of 2^31.

- MATLAB, octave and python interfaces do not handle single precision.
    
- Fortran interface does not allow control of options, nor data sizes exceeding 2^31.
  


Bug reports
***********
  
If you think you have found a bug, please
file an issue on the github project page,
https://github.com/flatironinstitute/finufft/issues .
Include a minimal code which reproduces the bug, along with
details about your machine, operating system, compiler, and version of FINUFFT.

You may also contact Alex Barnett (``abarnett``
at-sign ``flatironinstitute.org``) with FINUFFT in the subject line.

