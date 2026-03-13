// native: finufft
// wasm: finufft
// finufft_destroy(handle) -> void
register({
  check: function (argTypes, nargout) {
    return { outputTypes: [] };
  },
  apply: function (args, nargout) {
    var handle = args[0];
    if (native) {
      var fn = native.func("void guru_destroy(int handle)");
      fn(handle);
      return;
    }
    wasm.exports.guru_destroy(handle);
  },
});
