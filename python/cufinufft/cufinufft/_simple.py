from cufinufft import Plan, _compat

def nufft1d1(x, data, n_modes=None, out=None, eps=1e-6, isign=1, **kwargs):
    return _invoke_plan(1, 1, x, None, None, data, out, isign, eps, n_modes,
            kwargs)

def nufft1d2(x, data, out=None, eps=1e-6, isign=-1, **kwargs):
    return _invoke_plan(1, 2, x, None, None, data, out, isign, eps, None,
            kwargs)

def nufft2d1(x, y, data, n_modes=None, out=None, eps=1e-6, isign=1, **kwargs):
    return _invoke_plan(2, 1, x, y, None, data, out, isign, eps, n_modes,
            kwargs)

def nufft2d2(x, y, data, out=None, eps=1e-6, isign=-1, **kwargs):
    return _invoke_plan(2, 2, x, y, None, data, out, isign, eps, None, kwargs)

def nufft3d1(x, y, z, data, n_modes=None, out=None, eps=1e-6, isign=1,
        **kwargs):
    return _invoke_plan(3, 1, x, y, z, data, out, isign, eps, n_modes, kwargs)

def nufft3d2(x, y, z, data, out=None, eps=1e-6, isign=-1, **kwargs):
    return _invoke_plan(3, 2, x, y, z, data, out, isign, eps, None, kwargs)

def _invoke_plan(dim, nufft_type, x, y, z, data, out, isign, eps,
        n_modes=None, kwargs=None):
    dtype = _compat.get_array_dtype(data)

    n_trans = _get_ntrans(dim, nufft_type, data)

    if nufft_type == 1 and out is not None:
        n_modes = out.shape[-dim:]
    if nufft_type == 2:
        n_modes = data.shape[-dim:]

    plan = Plan(nufft_type, n_modes, n_trans, eps, isign, dtype, **kwargs)

    plan.setpts(x, y, z)

    if out is None:
        out = plan.execute(data)
    else:
        plan.execute(data, out=out)

    return out


def _get_ntrans(dim, nufft_type, data):
    if nufft_type == 1:
        expect_dim = 1
    else:
        expect_dim = dim

    if data.ndim == expect_dim:
        n_trans = 1
    else:
        n_trans = data.shape[0]

    return n_trans


def _wrap_docstring(docstring, tw=80, min_spacing=2):
    lines = docstring.expandtabs().splitlines()

    for k, line in enumerate(lines):
        if len(line) > tw:
            last_space = line[:tw].rfind(' ')
            indent_level = line.rfind(' ' * min_spacing) + min_spacing

            lines[k] = line[:last_space]

            new_line = (' ' * indent_level) + line[last_space + 1:]

            # Check if the indentation level continues on next line. If so,
            # concatenate, otherwise insert new line.
            if len(lines[k + 1]) - len(lines[k + 1].lstrip()) >= indent_level:
                lines[k + 1] = new_line + ' ' + lines[k + 1].lstrip()
            else:
                lines.insert(k + 1, new_line)

    docstring = '\n'.join(lines)

    return docstring


def _set_nufft_doc(f, dim, tp):
    doc_nufft1 = \
    """{dim}D type-1 (nonuniform to uniform) complex NUFFT

    ::

      {pt_spacing}        M-1
      f[{pt_idx}] = SUM c[j] exp(+/-i {pt_inner})
      {pt_spacing}        j=0

          for {pt_constraint}

    Args:
{pts_doc}
      c         (complex[M] or complex[n_tr, M]): source strengths.
      n_modes   (integer or integer tuple of length {dim}, optional): number of
                uniform Fourier modes requested {modes_tuple}. May be even or odd; in
                either case, modes {pt_idx} are integers satisfying {pt_constraint}.
                Must be specified if ``out`` is not given.
      out       (complex[{modes}] or complex[n_tr, {modes}], optional): output array
                for Fourier mode values. If ``n_modes`` is specifed, the shape
                must match, otherwise ``n_modes`` is inferred from ``out``.
      eps       (float, optional): precision requested (>1e-16).
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      **kwargs  (optional): other options may be specified, see the ``Plan``
                constructor for details.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[{modes}] or complex[n_tr, {modes}]: The resulting array.

    Example (CuPy):
    ::

      import cupy as cp
      import cufinufft

      # number of nonuniform points
      M = 100

      # the nonuniform points
{pts_generate}

      # their complex strengths
      c = (cp.random.standard_normal(size=M)
           + 1J * cp.random.standard_normal(size=M))

      # desired number of Fourier modes
      {modes} = {sample_modes}

      # calculate the type-1 NUFFT
      f = cufinufft.nufft{dim}d1({pts}, c, {modes_tuple})
    """

    doc_nufft2 = \
    """{dim}D type-2 (uniform to nonuniform) complex NUFFT

    ::

      c[j] = SUM f[{pt_idx}] exp(+/-i {pt_inner})
             {pt_idx}

          for j = 0, ..., M-1, where the sum is over {pt_constraint}

    Args:
{pts_doc}
      f         (complex[{modes}] or complex[n_tr, {modes}]): Fourier mode
                coefficients, where {modes} may be even or odd. In either case
                the mode indices {pt_idx} satisfy {pt_constraint}.
      out       (complex[M] or complex[n_tr, M], optional): output array
                at targets.
      eps       (float, optional): precision requested (>1e-16).
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      **kwargs  (optional): other options may be specified, see the ``Plan``
                constructor for details.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[M] or complex[n_tr, M]: The resulting array.

    Example (CuPy):
    ::

      import cupy as cp
      import cufinufft

      # number of nonuniform points
      M = 100

      # the nonuniform points
{pts_generate}

      # number of Fourier modes
      {modes} = {sample_modes}

      # the Fourier mode coefficients
      f = (cp.random.standard_normal(size={modes_tuple})
           + 1J * cp.random.standard_normal(size={modes_tuple}))

      # calculate the type-2 NUFFT
      c = cufinufft.nufft{dim}d2({pts}, f)
    """

    doc_nufft = {1: doc_nufft1, 2: doc_nufft2}

    pts = ('x', 'y', 'z')
    sample_modes = (50, 75, 100)

    dims = range(1, dim + 1)

    v = {}

    v['dim'] = dim

    v['modes'] = ', '.join('N{}'.format(i) for i in dims)
    v['modes_tuple'] = '(' + v['modes'] + (', ' if dim == 1 else '') + ')'
    v['pt_idx'] = ', '.join('k{}'.format(i) for i in dims)
    v['pt_spacing'] = ' ' * (len(v['pt_idx']) - 2)
    v['pt_inner'] = ' + '.join('k{0} {1}(j)'.format(i, x) for i, x in zip(dims, pts[:dim]))
    v['pt_constraint'] = ', '.join('-N{0}/2 <= k{0} <= (N{0}-1)/2'.format(i) for i in dims)
    v['pts_doc'] = '\n'.join('      {}         (float[M]): nonuniform points, in the interval [-pi, pi), values outside will be folded'.format(x) for x in pts[:dim])

    # for example
    v['pts'] = ', '.join(str(x) for x in pts[:dim])
    v['pts_generate'] = '\n'.join('      {} = 2 * cp.pi * cp.random.uniform(size=M)'.format(x) for x in pts[:dim])
    v['sample_modes'] = ', '.join(str(n) for n in sample_modes[:dim])

    if dim > 1:
        v['pt_inner'] = '(' + v['pt_inner'] + ')'

    f.__doc__ = _wrap_docstring(doc_nufft[tp].format(**v))


_set_nufft_doc(nufft1d1, 1, 1)
_set_nufft_doc(nufft1d2, 1, 2)
_set_nufft_doc(nufft2d1, 2, 1)
_set_nufft_doc(nufft2d2, 2, 2)
_set_nufft_doc(nufft3d1, 3, 1)
_set_nufft_doc(nufft3d2, 3, 2)
