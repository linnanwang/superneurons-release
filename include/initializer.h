//
// Created by ay27 on 7/11/17.
//

#ifndef SUPERNEURONS_INITIALIZER_H
#define SUPERNEURONS_INITIALIZER_H

#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <cassert>


typedef enum INIT_TYPE {
    _sequential=0,
    _constant = 1,
    _random = 2,
    _gaussian = 3,
    _xavier = 4,
    _variance = 5
}INIT_TYPE;

namespace SuperNeurons {

template<class value_type>
class initializer_t {
public:
    virtual void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) = 0;
    virtual INIT_TYPE get_type() = 0;
};


template<class value_type>
class sequential_initializer_t : public initializer_t<value_type> {
public:
    sequential_initializer_t() {}

    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override;

    INIT_TYPE get_type() {
        return _sequential;
    }
};

template<class value_type>
class constant_initializer_t : public initializer_t<value_type> {
private:
    value_type const_val;
public:
    constant_initializer_t(value_type _const_value) : const_val(_const_value) {}

    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override {
        long total = N * C * H * W;
        assert(cpu_ptr != NULL);
        for (int i = 0; i < total; i++) {
            cpu_ptr[i] = const_val;
            //if(i % 20 == 0) cpu_ptr[i] = i * const_val;
            //else           cpu_ptr[i] = 0;
        }
    }

    INIT_TYPE get_type() {
        return _constant;
    }
};


template<class value_type>
class random_initializer_t : public initializer_t<value_type> {

public:
    random_initializer_t() {}

    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override;

    INIT_TYPE get_type() {
        return _random;
    }
};


template<class value_type>
class gaussian_initializer_t : public initializer_t<value_type> {
private:
    value_type mean, std;
public:
    gaussian_initializer_t(value_type _mean, value_type _std) : mean(_mean), std(_std) {}

    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override;

    INIT_TYPE get_type() {
        return _gaussian;
    }
};


typedef enum fan_type_t {
    FAN_IN = 0,
    FAN_OUT = 1,
    FAN_AVG = 2,
} fan_type_t;

template<class value_type>
class variance_scaling_initializer_t : public initializer_t<value_type> {
private:
    fan_type_t type;
    value_type factor;
    bool uniform;
public:
    variance_scaling_initializer_t(fan_type_t _type, value_type _factor, bool _uniform = false)
            : type(_type), factor(_factor), uniform(_uniform) {}

    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override;

    INIT_TYPE get_type() {
        return _variance;
    }
};

template<class value_type>
class xavier_initializer_t : public initializer_t<value_type> {
public:
    void call(value_type *cpu_ptr, size_t N, size_t C, size_t H, size_t W) override;

    INIT_TYPE get_type() {
        return _xavier;
    }
};


} // namespace SuperNeurons


#endif //SUPERNEURONS_INITIALIZER_H
