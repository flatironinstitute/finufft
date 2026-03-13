// wasm: finufft
// finufft_setpts(handle, nj, xj, yj, zj, nk, s, t, u) -> void
// Sets nonuniform points on a guru plan.
register({
  check: function (argTypes, nargout) {
    return { outputTypes: [] };
  },
  apply: function (args, nargout) {
    var handle = args[0];
    var nj = args[1];
    var xj = args[2]; // tensor or empty
    var yj = args[3]; // tensor or empty
    var zj = args[4]; // tensor or empty
    var nk = args[5];
    var s = args[6]; // tensor or empty
    var t = args[7]; // tensor or empty
    var u = args[8]; // tensor or empty

    var BYTES = 8;
    var exports = wasm.exports;
    var mem = exports.memory;

    function allocCopy(tensor, n) {
      if (n === 0 || !tensor || !tensor.data || tensor.data.length === 0) {
        return 0; // null pointer for empty arrays
      }
      var ptr = exports.my_malloc(n * BYTES);
      var view = new Float64Array(mem.buffer);
      view.set(new Float64Array(tensor.data.buffer, tensor.data.byteOffset, n), ptr / BYTES);
      return ptr;
    }

    var xj_ptr = allocCopy(xj, nj);
    var yj_ptr = allocCopy(yj, nj);
    var zj_ptr = allocCopy(zj, nj);
    var s_ptr = allocCopy(s, nk);
    var t_ptr = allocCopy(t, nk);
    var u_ptr = allocCopy(u, nk);

    var ier = exports.guru_setpts(handle, nj, xj_ptr, yj_ptr, zj_ptr,
                                  nk, s_ptr, t_ptr, u_ptr);

    if (xj_ptr) exports.my_free(xj_ptr);
    if (yj_ptr) exports.my_free(yj_ptr);
    if (zj_ptr) exports.my_free(zj_ptr);
    if (s_ptr) exports.my_free(s_ptr);
    if (t_ptr) exports.my_free(t_ptr);
    if (u_ptr) exports.my_free(u_ptr);

    if (ier !== 0) {
      throw new RuntimeError("finufft_setpts failed with error code " + ier);
    }
  },
});
