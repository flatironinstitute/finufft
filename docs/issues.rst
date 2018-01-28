Known Issues
============

One should also check the github issues for the project page,
https://github.com/ahbarnett/finufft/issues

Also see notes in the ``TODO`` file.

Issues with library
*******************

- When requestes accuracy is ``1e-14`` or less, it is sometimes not possible to match this, especially when there are a large number of input and/or output points. This is believed to be unavoidable round-off error.

- Currently in Mac OSX, ``make lib`` fails to make the shared object library (.so).

- The timing of the first FFTW call is complicated, depending on whether FFTW_ESTIMATE (the default) or FFTW_MEASURE is used. Such issues are known, and discussed in other documentation, eg https://pythonhosted.org/poppy/fft_optimization.html
  We would like to find a way of pre-storing some intel FFTW plans (as MATLAB does) to avoid the large FFTW_ESTIMATE planning time.
  

Issues with interfaces
**********************

- MATLAB, octave and python cannot exceed input or output data sizes of 2^31.

- A segfault occurs if MATLAB's ``fft`` is called before the first ``finufft``
  call in a session.
  We believe this due to incompatibility between the versions of
  FFTW used. Please contact us if you know of a fix.

  Workaround: in your ``startup.m`` file, include a dummy call as follows::

    finufft1d1(1,1,1,1,1);

  This issue does not occur with octave.

 

Bug reports
***********
  
If you think you have found a bug, please
file an issue on the github project page,
https://github.com/ahbarnett/finufft/issues
Include a minimal code which reproduces the bug, along with
details about your machine, operating system, compiler, and version of FINUFFT.

You may also contact Alex Barnett (``abarnett``
at-sign ``flatironinstitute.org``) with FINUFFT in the subject line.

