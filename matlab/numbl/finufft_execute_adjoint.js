// native: finufft
// wasm: finufft
// finufft_execute_adjoint(handle, data_in, n_in, n_out) -> complex_tensor
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "tensor" }],
      apply: function (args, nargout) {
        var handle = args[0];
        var data_in = args[1];
        var n_in = args[2];
        var n_out = args[3];

        // Coerce scalar/complex to tensor if needed
        if (typeof data_in === "number") {
          var scalar_re = new FloatXArray(1);
          scalar_re[0] = data_in;
          data_in = { data: scalar_re, imag: null, shape: [1, 1] };
        } else if (data_in && data_in.kind === "complex_number") {
          var cplx_re = new FloatXArray(1);
          var cplx_im = new FloatXArray(1);
          cplx_re[0] = data_in.re;
          cplx_im[0] = data_in.im;
          data_in = { data: cplx_re, imag: cplx_im, shape: [1, 1] };
        }

        if (native) {
          var fn = native.func("int guru_execute_adjoint(int handle, double *in_re, double *in_im, int n_in, double *out_re, double *out_im, int n_out)");

          var in_re = new Float64Array(n_in);
          var in_im = new Float64Array(n_in);
          for (var i = 0; i < n_in; i++) {
            in_re[i] = data_in.data[i];
            in_im[i] = data_in.imag ? data_in.imag[i] : 0;
          }
          var out_re = new Float64Array(n_out);
          var out_im = new Float64Array(n_out);

          var ier = fn(handle, in_re, in_im, n_in, out_re, out_im, n_out);
          if (ier !== 0) {
            throw new RuntimeError("finufft_execute_adjoint failed with error code " + ier);
          }

          var result_re = new FloatXArray(n_out);
          var result_im = new FloatXArray(n_out);
          for (var i = 0; i < n_out; i++) {
            result_re[i] = out_re[i];
            result_im[i] = out_im[i];
          }
          return RTV.tensor(result_re, [n_out, 1], result_im);
        }

        var BYTES = 8;
        var exports = wasm.exports;
        var mem = exports.memory;

        var in_re_ptr = exports.my_malloc(n_in * BYTES);
        var in_im_ptr = exports.my_malloc(n_in * BYTES);
        var out_re_ptr = exports.my_malloc(n_out * BYTES);
        var out_im_ptr = exports.my_malloc(n_out * BYTES);

        var view = new Float64Array(mem.buffer);
        view.set(new Float64Array(data_in.data.buffer, data_in.data.byteOffset, n_in), in_re_ptr / BYTES);
        if (data_in.imag) {
          view.set(new Float64Array(data_in.imag.buffer, data_in.imag.byteOffset, n_in), in_im_ptr / BYTES);
        } else {
          view.fill(0, in_im_ptr / BYTES, in_im_ptr / BYTES + n_in);
        }

        var ier = exports.guru_execute_adjoint(handle, in_re_ptr, in_im_ptr, n_in,
                                               out_re_ptr, out_im_ptr, n_out);

        view = new Float64Array(mem.buffer);
        var out_re = new FloatXArray(n_out);
        var out_im = new FloatXArray(n_out);
        for (var i = 0; i < n_out; i++) {
          out_re[i] = view[out_re_ptr / BYTES + i];
          out_im[i] = view[out_im_ptr / BYTES + i];
        }

        exports.my_free(in_re_ptr);
        exports.my_free(in_im_ptr);
        exports.my_free(out_re_ptr);
        exports.my_free(out_im_ptr);

        if (ier !== 0) {
          throw new RuntimeError("finufft_execute_adjoint failed with error code " + ier);
        }

        return RTV.tensor(out_re, [n_out, 1], out_im);
      },
    };
  },
});
