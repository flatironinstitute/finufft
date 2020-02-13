// JFM's class for passing arrays between C++ and python

#ifndef ndarray_h
#define ndarray_h

#include <pybind11/numpy.h>
namespace py = pybind11;

template <typename T>
struct NDArray {
	NDArray(py::array_t<T> &X) {
		auto buf=X.request();
		ndim=buf.ndim;
		size=buf.size;
		for (int d=0; d<ndim; d++) {
			shape.push_back(buf.shape[d]);
		}
		ptr=(T*)buf.ptr;
	}

	long int ndim=0; // The number of dimensions
	long int size=0; // The total size of the array (product of dimensions)
	std::vector<int> shape; // The sizes of the dimensions
	T *ptr=0; // Pointer to the actual data
};

#endif
