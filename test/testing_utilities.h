#pragma once

// Common functions used for testing

#include <complex>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>


namespace finufft {
    namespace detail {
        template<typename T>
        struct type_identity {
            typedef T type;
        };
    }

    // Fills the given array with random data.
    // The distribution is uniform in the range [min, max].
    // Note: the slightly awkward signature is to avoid having the min and max arguments be template type-deduced,
    //     and instead be automatically converted to the type of the array.
    template<typename T>
    void fill_random(T* data, size_t size, int seed, typename detail::type_identity<T>::type min, typename detail::type_identity<T>::type max) {
        typedef r123::Philox2x32 RNG;
        RNG rng;

        RNG::ctr_type ctr = {{}};
        RNG::ukey_type key = {{}};
        key[0] = seed;

        for (size_t i = 0; i < size; ++i) {
            ctr[0] = i;
            auto r = rng(ctr, key);
            data[i] = min + r123::u01<T>(r[0]) * (max - min);
        }
    }

    // Fills the given array with random data.
    // The distribution is uniform in the square [min, max]^2 in complex domain.
    template<typename T>
    void fill_random(std::complex<T>* data, size_t size, int seed, typename detail::type_identity<T>::type min, typename detail::type_identity<T>::type max) {
        typedef r123::Philox2x32 RNG;
        RNG rng;

        RNG::ctr_type ctr = {{}};
        RNG::ukey_type key = {{}};
        key[0] = seed;

        for (size_t i = 0; i < size; ++i) {
            ctr[0] = i;
            auto r = rng(ctr, key);
            data[i].real(min + r123::u01<T>(r[0]) * (max - min));
            data[i].imag(min + r123::u01<T>(r[1]) * (max - min));
        }
    }
}
