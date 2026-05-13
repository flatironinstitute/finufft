// native: finufft
// wasm: finufft
//
// Single-builtin replacement for the upstream finufft MEX file.
//
// The upstream `finufft_plan.m`, `finufft1d1.m`, etc. all dispatch through
// `finufft(mex_id_, ...args)` where `mex_id_` is the mwrap-generated stub
// signature string (e.g. 'c o int = finufft_setpts(...)').  This builtin
// marshals each runtime arg into a shim mxArray, calls our wasm/native
// `mex_dispatch` (which forwards to the original mexFunction inside
// finufft.cpp), and decodes the resulting plhs[] back into runtime values.
//
// Class instances (the finufft_plan classdef object) are unwrapped to their
// `mwptr` string field on the JS side so the cpp code's mxIsChar/sscanf
// branch handles them — no R2008OO / mxGetProperty needed in the shim.

register({
  resolve: function (argTypes, nargout) {
    if (argTypes.length < 1) return null;
    var outs = [];
    var n = nargout > 0 ? nargout : 0;
    for (var i = 0; i < n; i++) outs.push({ kind: "unknown" });
    return {
      outputTypes: outs,
      apply: function (args, nargout) {
        return callFinufft(args, nargout);
      },
    };
  },
});

// ── runtime-value helpers ──────────────────────────────────────────────────

function isTensor(v) {
  return v && typeof v === "object" && v.kind === "tensor";
}
function isChar(v) {
  return v && typeof v === "object" && v.kind === "char";
}
function isStruct(v) {
  return v && typeof v === "object" && v.kind === "struct";
}
function isClassInstance(v) {
  return v && typeof v === "object" && v.kind === "class_instance";
}
function isComplexNumber(v) {
  return v && typeof v === "object" && v.kind === "complex_number";
}

function tensorElemCount(t) {
  var n = 1;
  for (var i = 0; i < t.shape.length; i++) n *= t.shape[i];
  return n;
}

// MATLAB-style m and n for an mxArray reflecting the runtime tensor.
// Numbl tensors keep at least 2D shape; we collapse trailing singletons
// the same way mwrap does (m * n must equal numel).
function tensorMxDims(t) {
  var s = t.shape;
  var m = s[0] | 0;
  var n = 1;
  for (var i = 1; i < s.length; i++) n *= s[i];
  return { m: m, n: n };
}

// ── mxArray construction (delegates to bridge for the buffer transport) ───

function buildMxArray(bridge, v) {
  if (typeof v === "number") {
    return bridge.makeDoubleScalar(v);
  }
  if (typeof v === "boolean") {
    return bridge.makeDoubleScalar(v ? 1 : 0);
  }
  if (typeof v === "string") {
    return bridge.makeString(v);
  }
  if (isChar(v)) {
    return bridge.makeString(v.value || "");
  }
  if (isComplexNumber(v)) {
    var rer = new Float64Array(1);
    var imr = new Float64Array(1);
    rer[0] = v.re;
    imr[0] = v.im;
    return bridge.makeComplexMatrix(1, 1, rer, imr);
  }
  if (isTensor(v)) {
    var dims = tensorMxDims(v);
    var n = dims.m * dims.n;
    if (v.imag) {
      var re = n === v.data.length ? v.data : v.data.subarray(0, n);
      var im = n === v.imag.length ? v.imag : v.imag.subarray(0, n);
      return bridge.makeComplexMatrix(dims.m, dims.n,
                                      asFloat64(re), asFloat64(im));
    }
    var data = n === v.data.length ? v.data : v.data.subarray(0, n);
    return bridge.makeRealMatrix(dims.m, dims.n, asFloat64(data));
  }
  if (isStruct(v)) {
    var keys = [];
    var values = [];
    v.fields.forEach(function (val, key) {
      keys.push(key);
      values.push(val);
    });
    var s = bridge.makeStruct(keys.length);
    for (var i = 0; i < keys.length; i++) {
      var sub = buildMxArray(bridge, values[i]);
      bridge.structSetField(s, i, keys[i], sub);
    }
    return s;
  }
  if (isClassInstance(v)) {
    // Unwrap classdef objects (finufft_plan) — pass mwptr string field through.
    var mw = v.fields.get("mwptr");
    if (mw === undefined) {
      throw new RuntimeError("finufft: class instance has no mwptr field");
    }
    return buildMxArray(bridge, mw);
  }
  throw new RuntimeError("finufft: unsupported argument type " + (v && v.kind));
}

function asFloat64(arr) {
  if (arr instanceof Float64Array) return arr;
  // Numbl may use Float32Array if NUMBL_USE_FLOAT32 is set; convert.
  return new Float64Array(arr);
}

// ── mxArray decoding back to runtime values ───────────────────────────────

function decodeMxArray(bridge, mx) {
  var classID = bridge.getClassID(mx);
  var m = bridge.getM(mx);
  var n = bridge.getN(mx);
  var isCpx = bridge.getIsComplex(mx);

  // mxClassID values match the enum in mex_shim/mex.h
  var MX_CHAR_CLASS = 4;
  var MX_DOUBLE_CLASS = 6;
  var MX_STRUCT_CLASS = 2;

  if (classID === MX_CHAR_CLASS) {
    return RTV.char(bridge.readString(mx));
  }
  if (classID === MX_DOUBLE_CLASS) {
    var total = m * n;
    if (total === 1 && !isCpx) {
      return RTV.num(bridge.readDoubleScalar(mx));
    }
    if (isCpx) {
      var re = new FloatXArray(total);
      var im = new FloatXArray(total);
      bridge.readComplex(mx, total, re, im);
      return RTV.tensor(re, [m, n], im);
    }
    var data = new FloatXArray(total);
    bridge.readReal(mx, total, data);
    return RTV.tensor(data, [m, n]);
  }
  if (classID === MX_STRUCT_CLASS) {
    // No code path in finufft.cpp returns structs; fall through to error.
  }
  throw new RuntimeError("finufft: unsupported output mxArray classID " + classID);
}

// ── wasm bridge ────────────────────────────────────────────────────────────

function makeWasmBridge() {
  var exports = wasm.exports;

  function memView64() {
    return new Float64Array(exports.memory.buffer);
  }
  function mem8() {
    return new Uint8Array(exports.memory.buffer);
  }

  function copyDoublesIn(arr, n) {
    if (!arr || n === 0) return 0;
    var ptr = exports.my_malloc(n * 8);
    memView64().set(arr.subarray(0, n), ptr / 8);
    return ptr;
  }

  function copyStringIn(s) {
    var bytes = new TextEncoder().encode(s + "\0");
    var ptr = exports.my_malloc(bytes.length);
    mem8().set(bytes, ptr);
    return ptr;
  }

  function readDoublesOut(ptr, n) {
    if (n === 0) return new Float64Array(0);
    var view = memView64();
    return new Float64Array(view.subarray(ptr / 8, ptr / 8 + n));
  }

  function readStringFromBuf(ptr, len) {
    var bytes = mem8().subarray(ptr, ptr + len);
    return new TextDecoder().decode(bytes);
  }

  return {
    makeDoubleScalar: function (v) {
      return exports.mex_make_double_scalar(v);
    },
    makeRealMatrix: function (m, n, data) {
      var ptr = copyDoublesIn(data, m * n);
      var mx = exports.mex_make_real_matrix(m, n, ptr);
      if (ptr) exports.my_free(ptr);
      return mx;
    },
    makeComplexMatrix: function (m, n, re, im) {
      var rePtr = copyDoublesIn(re, m * n);
      var imPtr = copyDoublesIn(im, m * n);
      var mx = exports.mex_make_complex_matrix(m, n, rePtr, imPtr);
      if (rePtr) exports.my_free(rePtr);
      if (imPtr) exports.my_free(imPtr);
      return mx;
    },
    makeString: function (s) {
      var ptr = copyStringIn(s);
      var mx = exports.mex_make_string(ptr);
      exports.my_free(ptr);
      return mx;
    },
    makeStruct: function (nfields) {
      return exports.mex_make_struct(nfields);
    },
    structSetField: function (s, idx, name, value) {
      var ptr = copyStringIn(name);
      exports.mex_struct_set_field(s, idx, ptr, value);
      exports.my_free(ptr);
    },

    getClassID: function (mx) { return exports.mex_get_classid(mx); },
    getM: function (mx) { return exports.mex_get_m(mx); },
    getN: function (mx) { return exports.mex_get_n(mx); },
    getIsComplex: function (mx) { return exports.mex_get_is_complex(mx); },

    readDoubleScalar: function (mx) {
      return exports.mex_read_double_scalar(mx);
    },
    readReal: function (mx, n, out) {
      var ptr = exports.my_malloc(n * 8);
      exports.mex_read_real(mx, ptr);
      var data = readDoublesOut(ptr, n);
      out.set(data);
      exports.my_free(ptr);
    },
    readComplex: function (mx, n, outRe, outIm) {
      var rePtr = exports.my_malloc(n * 8);
      var imPtr = exports.my_malloc(n * 8);
      exports.mex_read_complex(mx, rePtr, imPtr);
      outRe.set(readDoublesOut(rePtr, n));
      outIm.set(readDoublesOut(imPtr, n));
      exports.my_free(rePtr);
      exports.my_free(imPtr);
    },
    readString: function (mx) {
      var bufLen = exports.mex_get_m(mx) * exports.mex_get_n(mx) + 1;
      var ptr = exports.my_malloc(bufLen);
      var len = exports.mex_read_string(mx, ptr, bufLen);
      var s = readStringFromBuf(ptr, len);
      exports.my_free(ptr);
      return s;
    },

    allocArgs: function (n) { return exports.mex_alloc_args(n); },
    setArg: function (arr, idx, mx) { exports.mex_set_arg(arr, idx, mx); },
    getArg: function (arr, idx) { return exports.mex_get_arg(arr, idx); },
    freeArgs: function (arr) { exports.mex_free_args(arr); },
    freeArray: function (mx) { if (mx) exports.mex_free_array(mx); },
    dispatch: function (nlhs, plhs, nrhs, prhs) {
      return exports.mex_dispatch(nlhs, plhs, nrhs, prhs);
    },
    getError: function () {
      // mex_get_error returns a const char*. Walk wasm memory until NUL.
      var ptr = exports.mex_get_error();
      var bytes = mem8();
      var end = ptr;
      while (bytes[end] !== 0 && end - ptr < 4096) end++;
      return new TextDecoder().decode(bytes.subarray(ptr, end));
    },
  };
}

// ── native (koffi) bridge ──────────────────────────────────────────────────

var nativeFns = null;

function getNativeFns() {
  if (nativeFns) return nativeFns;
  var lib = native;
  nativeFns = {
    mex_make_double_scalar: lib.func("void *mex_make_double_scalar(double v)"),
    mex_make_real_matrix:   lib.func("void *mex_make_real_matrix(int m, int n, double *data)"),
    mex_make_complex_matrix:lib.func("void *mex_make_complex_matrix(int m, int n, double *re, double *im)"),
    mex_make_string:        lib.func("void *mex_make_string(const char *s)"),
    mex_make_struct:        lib.func("void *mex_make_struct(int n)"),
    mex_struct_set_field:   lib.func("void mex_struct_set_field(void *s, int idx, const char *name, void *val)"),
    mex_get_classid:        lib.func("int mex_get_classid(void *a)"),
    mex_get_m:              lib.func("int mex_get_m(void *a)"),
    mex_get_n:              lib.func("int mex_get_n(void *a)"),
    mex_get_is_complex:     lib.func("int mex_get_is_complex(void *a)"),
    mex_read_double_scalar: lib.func("double mex_read_double_scalar(void *a)"),
    mex_read_real:          lib.func("void mex_read_real(void *a, _Out_ double *out)"),
    mex_read_complex:       lib.func("void mex_read_complex(void *a, _Out_ double *out_re, _Out_ double *out_im)"),
    mex_read_string:        lib.func("int mex_read_string(void *a, _Out_ char *out, int buflen)"),
    mex_alloc_args:         lib.func("void **mex_alloc_args(int n)"),
    mex_set_arg:            lib.func("void mex_set_arg(void **arr, int idx, void *val)"),
    mex_get_arg:            lib.func("void *mex_get_arg(void **arr, int idx)"),
    mex_free_args:          lib.func("void mex_free_args(void **arr)"),
    mex_free_array:         lib.func("void mex_free_array(void *a)"),
    mex_dispatch:           lib.func("int mex_dispatch(int nlhs, void **plhs, int nrhs, void **prhs)"),
    mex_get_error:          lib.func("const char *mex_get_error()"),
  };
  return nativeFns;
}

function makeNativeBridge() {
  var fns = getNativeFns();
  return {
    makeDoubleScalar: function (v) { return fns.mex_make_double_scalar(v); },
    makeRealMatrix: function (m, n, data) {
      return fns.mex_make_real_matrix(m, n, m * n > 0 ? data : null);
    },
    makeComplexMatrix: function (m, n, re, im) {
      return fns.mex_make_complex_matrix(m, n,
                                         m * n > 0 ? re : null,
                                         m * n > 0 ? im : null);
    },
    makeString: function (s) { return fns.mex_make_string(s); },
    makeStruct: function (nfields) { return fns.mex_make_struct(nfields); },
    structSetField: function (s, idx, name, value) {
      fns.mex_struct_set_field(s, idx, name, value);
    },

    getClassID: function (mx) { return fns.mex_get_classid(mx); },
    getM: function (mx) { return fns.mex_get_m(mx); },
    getN: function (mx) { return fns.mex_get_n(mx); },
    getIsComplex: function (mx) { return fns.mex_get_is_complex(mx); },

    readDoubleScalar: function (mx) { return fns.mex_read_double_scalar(mx); },
    readReal: function (mx, n, out) {
      // koffi marshals JS typed arrays through `_Out_ double*` parameters
      var buf = out instanceof Float64Array ? out : new Float64Array(n);
      fns.mex_read_real(mx, buf);
      if (buf !== out) out.set(buf);
    },
    readComplex: function (mx, n, outRe, outIm) {
      var bufR = outRe instanceof Float64Array ? outRe : new Float64Array(n);
      var bufI = outIm instanceof Float64Array ? outIm : new Float64Array(n);
      fns.mex_read_complex(mx, bufR, bufI);
      if (bufR !== outRe) outRe.set(bufR);
      if (bufI !== outIm) outIm.set(bufI);
    },
    readString: function (mx) {
      var len = fns.mex_get_m(mx) * fns.mex_get_n(mx);
      var buf = new Uint8Array(len + 1);
      fns.mex_read_string(mx, buf, buf.length);
      // Trim trailing NUL
      var end = 0;
      while (end < len && buf[end] !== 0) end++;
      return new TextDecoder().decode(buf.subarray(0, end));
    },

    allocArgs: function (n) { return fns.mex_alloc_args(n); },
    setArg: function (arr, idx, mx) { fns.mex_set_arg(arr, idx, mx); },
    getArg: function (arr, idx) { return fns.mex_get_arg(arr, idx); },
    freeArgs: function (arr) { fns.mex_free_args(arr); },
    freeArray: function (mx) { if (mx) fns.mex_free_array(mx); },
    dispatch: function (nlhs, plhs, nrhs, prhs) {
      return fns.mex_dispatch(nlhs, plhs, nrhs, prhs);
    },
    getError: function () { return fns.mex_get_error(); },
  };
}

// ── per-plan setpts retention ─────────────────────────────────────────────
//
// finufft_setpts stores the xj/yj/zj/s/t/u pointers it is given without
// copying.  In real MATLAB those buffers stay alive because MATLAB owns
// prhs[] for the duration of the variable.  In our shim each call allocates
// fresh mxArrays in the C heap, so we'd free them on return and finufft
// would later dereference dangling pointers.
//
// Solution: when we see a setpts call, retain the input mxArrays under a
// per-plan key (the plan's mwptr string).  When we see a destroy call,
// free the retained mxArrays.  Subsequent setpts calls for the same plan
// also drop the previous retention.

var retainedByPlan = new Map();
var cachedBridge = null;

function getBridge() {
  if (cachedBridge) return cachedBridge;
  cachedBridge = native ? makeNativeBridge() : makeWasmBridge();
  return cachedBridge;
}

function planKeyOf(arg) {
  if (typeof arg === "string") return arg;
  if (isChar(arg)) return arg.value;
  if (isClassInstance(arg)) {
    var mw = arg.fields.get("mwptr");
    if (mw === undefined) return null;
    if (typeof mw === "string") return mw;
    if (isChar(mw)) return mw.value;
  }
  return null;
}

function getMexIdString(arg) {
  if (typeof arg === "string") return arg;
  if (isChar(arg)) return arg.value;
  return "";
}

// ── main entry ─────────────────────────────────────────────────────────────

function callFinufft(args, nargout) {
  var bridge = getBridge();

  var nrhs = args.length;
  // Cap nlhs at 2 — finufft.cpp's stubs never set more than two outputs.
  var nlhs = Math.max(nargout | 0, 0);
  if (nlhs > 2) nlhs = 2;
  var plhsSlots = 2;

  var mexId = getMexIdString(args[0]);
  var isSetpts = mexId.indexOf("= finufft_setpts(") !== -1 ||
                 mexId.indexOf("= finufftf_setpts(") !== -1;
  var isDestroy = mexId.indexOf("finufft_destroy(") === 0 ||
                  mexId.indexOf("finufftf_destroy(") === 0;
  var planKey = (isSetpts || isDestroy) && args.length > 1
    ? planKeyOf(args[1])
    : null;

  var prhs = bridge.allocArgs(nrhs);
  var plhs = bridge.allocArgs(plhsSlots);
  var ownedInputs = new Array(nrhs);
  var retainInputs = false;

  try {
    for (var i = 0; i < nrhs; i++) {
      var mx = buildMxArray(bridge, args[i]);
      ownedInputs[i] = mx;
      bridge.setArg(prhs, i, mx);
    }

    var rc = bridge.dispatch(plhsSlots, plhs, nrhs, prhs);
    if (rc !== 0) {
      throw new RuntimeError("finufft: " + bridge.getError());
    }

    if (isSetpts && planKey) {
      // Drop any previous retention for this plan, then keep the current
      // input mxArrays alive.  finufft_setpts only stored pointers into
      // prhs[2..9] (nj, xj, yj, zj, nk, s, t, u), but it's simpler and
      // harmless to retain prhs[1..end].
      var prev = retainedByPlan.get(planKey);
      if (prev) {
        for (var p = 0; p < prev.length; p++) bridge.freeArray(prev[p]);
      }
      retainedByPlan.set(planKey, ownedInputs.slice(1));
      retainInputs = true;
    } else if (isDestroy && planKey) {
      var pr = retainedByPlan.get(planKey);
      if (pr) {
        for (var q = 0; q < pr.length; q++) bridge.freeArray(pr[q]);
        retainedByPlan.delete(planKey);
      }
    }

    if (nlhs === 0) return;
    if (nlhs === 1) {
      var out0 = bridge.getArg(plhs, 0);
      var v = decodeMxArray(bridge, out0);
      bridge.freeArray(out0);
      return v;
    }
    var results = [];
    for (var k = 0; k < nlhs; k++) {
      var mxOut = bridge.getArg(plhs, k);
      results.push(decodeMxArray(bridge, mxOut));
      bridge.freeArray(mxOut);
    }
    // Free any extra plhs slots written by the stub but not read by caller.
    for (var k2 = nlhs; k2 < plhsSlots; k2++) {
      var mxExtra = bridge.getArg(plhs, k2);
      if (mxExtra) bridge.freeArray(mxExtra);
    }
    return results;
  } finally {
    if (!retainInputs) {
      for (var j = 0; j < nrhs; j++) bridge.freeArray(ownedInputs[j]);
    } else {
      // mex_id_ (args[0]) is not retained because the C side only used it
      // for dispatch, not as a pointer source.
      bridge.freeArray(ownedInputs[0]);
    }
    bridge.freeArgs(prhs);
    bridge.freeArgs(plhs);
  }
}
