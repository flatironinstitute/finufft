// wasm: finufft
// finufft_makeplan(type, dim, n_modes, iflag, ntrans, tol) -> handle
// Creates a FINUFFT guru plan and returns an integer handle.
register({
  check: function (argTypes, nargout) {
    if (argTypes.length !== 6) {
      return null;
    }
    return { outputTypes: [IType.num()] };
  },
  apply: function (args, nargout) {
    var type = args[0];
    var dim = args[1];
    var n_modes = args[2]; // tensor [ms, mt, mu]
    var iflag = args[3];
    var ntrans = args[4];
    var tol = args[5];

    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    // Copy n_modes array to WASM (3 doubles)
    var nm_ptr = exports.my_malloc(3 * BYTES);
    var view = new Float64Array(mem.buffer);
    view.set(new Float64Array(n_modes.data.buffer, n_modes.data.byteOffset, 3), nm_ptr / BYTES);

    var handle = exports.guru_makeplan(type, dim, nm_ptr, iflag, ntrans, tol);
    exports.my_free(nm_ptr);

    if (handle < 0) {
      throw new RuntimeError("finufft_makeplan failed");
    }
    return RTV.num(handle);
  },
});
