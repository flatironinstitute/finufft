// native: finufft
// wasm: finufft
// finufft_makeplan(type, dim, n_modes, iflag, ntrans, tol) -> handle
register({
  resolve: function (argTypes, nargout) {
    if (argTypes.length !== 6) {
      return null;
    }
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args, nargout) {
        var type = args[0];
        var dim = args[1];
        var n_modes = args[2]; // tensor [ms, mt, mu]
        var iflag = args[3];
        var ntrans = args[4];
        var tol = args[5];

        if (native) {
          var fn = native.func("int guru_makeplan(int type, int dim, double *n_modes, int iflag, int ntrans, double tol)");
          var nm = new Float64Array(3);
          nm[0] = n_modes.data[0];
          nm[1] = n_modes.data.length > 1 ? n_modes.data[1] : 1;
          nm[2] = n_modes.data.length > 2 ? n_modes.data[2] : 1;
          var handle = fn(type, dim, nm, iflag, ntrans, tol);
          if (handle < 0) {
            throw new RuntimeError("finufft_makeplan failed");
          }
          return RTV.num(handle);
        }

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
    };
  },
});
