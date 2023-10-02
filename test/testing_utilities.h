#pragma once

// Common functions used for testing

#include <complex>
#include <random>

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
        std::random_device rd;
        std::mt19937 rng;
        rng.seed(seed);
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < size; ++i)
            data[i] = dist(rng);
    }

    // Fills the given array with random data.
    // The distribution is uniform in the square [min, max]^2 in complex domain.
    template<typename T>
    void fill_random(std::complex<T>* data, size_t size, int seed, typename detail::type_identity<T>::type min, typename detail::type_identity<T>::type max) {
        std::random_device rd;
        std::mt19937 rng;
        rng.seed(seed);
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < size; ++i) {
            data[i].real(dist(rng));
            data[i].imag(dist(rng));
        }
    }
}
