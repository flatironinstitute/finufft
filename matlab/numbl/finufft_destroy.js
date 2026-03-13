// wasm: finufft
// finufft_destroy(handle) -> void
// Destroys a FINUFFT guru plan and frees resources.
register({
  check: function (argTypes, nargout) {
    return { outputTypes: [] };
  },
  apply: function (args, nargout) {
    var handle = args[0];
    wasm.exports.guru_destroy(handle);
  },
});
