#include <nanobind/nanobind.h>

#include "finufft.h"

namespace nb = nanobind;


NB_MODULE(pyfinufft_ext, m) {
    nb::class_<finufft_opts>(m, "finufft_opts")
            .def(nb::init<>());
    m.def("finufft_default_opts", [](nb::capsule opts) {
        finufft_default_opts((finufft_opts*)opts.data());
    });
}