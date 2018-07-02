//
// Created by ay27 on 7/18/17.
//

#ifndef SUPERNEURONS_SOLVER_H
#define SUPERNEURONS_SOLVER_H

#include <cublas_alias.h>
#include <tensor.h>

namespace SuperNeurons {

typedef enum lr_decay_type {
    ITER = 0,
    LOSS = 1
} lr_decay_type;

typedef enum solver_type {
    MOMENTUM = 0,
    SGD = 1,
    NESTEROV = 2,
    ADAGRAD = 3,
    RMSPROP = 4
} solver_type;

template<class value_type>
class base_solver_t {
private:
    // record the iter locally, to update the lr correctly.
    size_t start_idx = 0;

    /**
     * the regularization function, will be called by update function.
     */
    void regularization(cublasHandle_t *handle,
                        tensor_t<value_type> *p,
                        tensor_t<value_type> *p_grad);

protected:
    solver_type my_type;
    value_type lr = 0.0;
    value_type weight_decay = 0.0;
    std::vector<std::pair<double, value_type> > lr_policy;
    lr_decay_type decay_type;

    /**
     * value_type: solver_type
     * value_type: lr
     * value_type: weight_decay
     * value_type: decay_type
     * value_type: policy_len
     * value_type: {{point, lr}, {point, lr}, ...}
     */
    void gen_base_description(char *dst, int *len_in_byte) {
        size_t SIZE = sizeof(value_type);
        value_type solver_type = this->my_type;
        memcpy(dst, &solver_type, SIZE);

        memcpy(dst + 1 * SIZE, &(this->lr), SIZE);
        memcpy(dst + 2 * SIZE, &(this->weight_decay), SIZE);
        memcpy(dst + 3 * SIZE, &(this->decay_type), SIZE);

        value_type policy_len = this->lr_policy.size();
        memcpy(dst + 4 * SIZE, &policy_len, SIZE);

        for (size_t i = 0; i < this->lr_policy.size(); ++i) {
            value_type l = this->lr_policy[i].first, r = this->lr_policy[i].second;
            memcpy(dst + 5 * SIZE + 2 * i * SIZE, &l, SIZE);
            memcpy(dst + 5 * SIZE + 2 * i * SIZE + 1, &r, SIZE);
        }

        *len_in_byte = (5 + this->lr_policy.size() * 2) * SIZE;
    }

public:

    base_solver_t(solver_type mtype, value_type lr, value_type weight_decay)
        : my_type(mtype), lr(lr), weight_decay(weight_decay) {
        lr_policy.clear();
    }

    virtual ~base_solver_t() {

    }

    virtual void gen_description(char *dst, int *len_in_byte) = 0;

    /**
     * set the lr decay policy, should feed data likes : {1, 100, 2000}, {0.1, 0.01, 0.001}.
     * */
    void set_lr_decay_policy(lr_decay_type _decay_type,
                             std::initializer_list<double> iters,
                             std::initializer_list<value_type> lrs);

    /**
     * update lr according to lr_policy.
     */
    void update_lr(size_t _iter, double avg_loss);

    value_type get_lr() {
        return lr;
    }

    /**
     * the main function of solver, will be called by network layer.
     * */
    void update(cublasHandle_t *handle,
                size_t _iter,
                tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev, bool enable_bias);

    /**
     * the main function implement by subclass.
     * */
    virtual void update_kernel(cublasHandle_t *handle,
                               tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                               tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                               tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev,
                               bool enable_bias) = 0;
};

template <typename value_type>
void momentum_update(int N, value_type* grad, value_type* prev, value_type momentum, value_type local_rate);

    
template<class value_type>
class momentum_solver_t : base_solver_t<value_type> {
public:
    momentum_solver_t(value_type _init_lr, value_type _weight_decay, value_type _m)
        : base_solver_t<value_type>(MOMENTUM, _init_lr, _weight_decay), m(_m) {}

    void update_kernel(cublasHandle_t *handle,
                       tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                       tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev,
                       bool enable_bias) override {
        calculate_update(handle, weight, weight_grad, weight_prev, false);

        if (enable_bias)
            calculate_update(handle, bias, bias_grad, bias_prev, true);
    }

    /**
     * mesh: base_description
     * value_type: param_cnt
     * value_type: m
     */
    void gen_description(char *dst, int *len_in_byte) {
        int base_len_in_byte;
        this->gen_base_description(dst, &base_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type param_cnt = 1;

        memcpy(dst + base_len_in_byte, &param_cnt, SIZE);
        memcpy(dst + base_len_in_byte + SIZE, &m, SIZE);

        *len_in_byte = base_len_in_byte + 2 * SIZE;
    }

private:
    value_type m;

    void calculate_update(cublasHandle_t *handle,
                          tensor_t<value_type> *p, tensor_t<value_type> *p_grad,
                          tensor_t<value_type> *p_prev,
                          bool is_bias);
};

template<class value_type>
class sgd_solver_t : base_solver_t<value_type> {
public:
    sgd_solver_t(value_type _lr, value_type weight_decay) : base_solver_t<value_type>(SGD, _lr, weight_decay) {}

    void update_kernel(cublasHandle_t *handle, tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                       tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev,
                       bool enable_bias) override {
        calculate_update(handle, weight, weight_grad);

        if (enable_bias)
            calculate_update(handle, bias, bias_grad);
    }

    /**
     * mesh: base_description
     * value_type: param_cnt
     */
    void gen_description(char *dst, int *len_in_byte) {
        int base_len_in_byte;
        this->gen_base_description(dst, &base_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type param_cnt = 0;

        memcpy(dst + base_len_in_byte, &param_cnt, SIZE);

        *len_in_byte = base_len_in_byte + SIZE;
    }

private:

    void calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p, tensor_t<value_type> *p_grad);

};

template<class value_type>
class nesterov_solver_t : base_solver_t<value_type> {
public:
    nesterov_solver_t(value_type _init_lr, value_type _weight_decay, value_type _m)
        : base_solver_t<value_type>(NESTEROV, _init_lr, _weight_decay), m(_m) {}

    void update_kernel(cublasHandle_t *handle, tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                       tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev,
                       bool enable_bias) override {
        calculate_update(handle, weight, weight_grad, weight_prev);

        if (enable_bias)
            calculate_update(handle, bias, bias_grad, bias_prev);
    }

    /**
     * mesh: base_description
     * value_type: param_cnt
     * value_type: m
     */
    void gen_description(char *dst, int *len_in_byte) {
        int base_len_in_byte;
        this->gen_base_description(dst, &base_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type param_cnt = 1;

        memcpy(dst + base_len_in_byte, &param_cnt, SIZE);
        memcpy(dst + base_len_in_byte + SIZE, &m, SIZE);

        *len_in_byte = base_len_in_byte + 2 * SIZE;
    }

private:
    value_type m;

    void calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p, tensor_t<value_type> *p_grad,
                          tensor_t<value_type> *p_prev);
};

template<class value_type>
class adagrad_solver_t : base_solver_t<value_type> {
private:
    value_type eps;

    void calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p, tensor_t<value_type> *p_grad,
                          tensor_t<value_type> *p_cache);

public:
    /**
     *
     * @param lr
     * @param weight_decay
     * @param _eps : smoothing term, typical value from 1e-4 ~ 1e-8.
     */
    adagrad_solver_t(value_type lr, value_type weight_decay, value_type _eps)
        : base_solver_t<value_type>(ADAGRAD, lr, weight_decay), eps(_eps) {}

    void update_kernel(cublasHandle_t *handle, tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                       tensor_t<value_type> *weight_cache, tensor_t<value_type> *bias_cache,
                       bool enable_bias) override {
        calculate_update(handle, weight, weight_grad, weight_cache);
        if (enable_bias) {
            calculate_update(handle, bias, bias_grad, bias_cache);
        }
    }

    /**
     * mesh: base_description
     * value_type: param_cnt
     * value_type: eps
     */
    void gen_description(char *dst, int *len_in_byte) {
        int base_len_in_byte;
        this->gen_base_description(dst, &base_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type param_cnt = 1;

        memcpy(dst + base_len_in_byte, &param_cnt, SIZE);
        memcpy(dst + base_len_in_byte + SIZE, &eps, SIZE);

        *len_in_byte = base_len_in_byte + 2 * SIZE;
    }
};

template<class value_type>
void adagrad_update(int N, value_type *p_grad, value_type *cache, value_type eps, value_type lr);

template<class value_type>
class rmsprop_solver_t : base_solver_t<value_type> {
private:
    value_type eps, rms_decay;

    void calculate_update(cublasHandle_t *handle,
                          tensor_t<value_type> *p, tensor_t<value_type> *p_grad, tensor_t<value_type> *p_cache);

public:
    /**
     *
     * @param lr : learning rate
     * @param weight_decay : weight decay for regularization
     * @param rms_decay : a parameter only in RMSProp, typical value are [0.9, 0.99, 0.999].
     * @param eps : smoothing term, typical value from 1e-4 ~ 1e-8.
     */
    rmsprop_solver_t(value_type lr, value_type weight_decay, value_type rms_decay, value_type eps)
        : base_solver_t<value_type>(RMSPROP, lr, weight_decay), rms_decay(rms_decay), eps(eps) {}

    void update_kernel(cublasHandle_t *handle, tensor_t<value_type> *weight, tensor_t<value_type> *bias,
                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                       tensor_t<value_type> *weight_cache, tensor_t<value_type> *bias_cache,
                       bool enable_bias) override {
        calculate_update(handle, weight, weight_grad, weight_cache);

        if (enable_bias) {
            calculate_update(handle, bias, bias_grad, bias_cache);
        }
    }

    /**
     * mesh: base_description
     * value_type: param_cnt
     * value_type: eps
     * value_type: rms_decay
     */
    void gen_description(char *dst, int *len_in_byte) {
        int base_len_in_byte;
        this->gen_base_description(dst, &base_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type param_cnt = 2;

        memcpy(dst + base_len_in_byte, &param_cnt, SIZE);
        memcpy(dst + base_len_in_byte + SIZE, &eps, SIZE);
        memcpy(dst + base_len_in_byte + 2 * SIZE, &rms_decay, SIZE);

        *len_in_byte = base_len_in_byte + 3 * SIZE;
    }
};

template<class value_type>
void rmsprop_update(int N,
                    value_type *p_grad,
                    value_type *p_cache,
                    value_type rms_decay,
                    value_type eps,
                    value_type lr);


//template<class value_type>
//class adam_solver_t : base_solver_t<value_type> {
//private:
//    value_type beta1, beta2;
//public:
//    adam_solver_t(value_type lr, value_type weight_decay,
//                  value_type beta1, value_type beta2)
//            : base_solver_t(lr, weight_decay),
//              beta1(beta1),
//              beta2(beta2) {}
//
//    void update_kernel(cublasHandle_t *handle, tensor_t <value_type> *weight, tensor_t <value_type> *bias,
//                       tensor_t <value_type> *weight_grad, tensor_t <value_type> *bias_grad,
//                       tensor_t <value_type> *weight_prev, tensor_t <value_type> *bias_prev,
//                       bool enable_bias) override {
//        // TODO : the Adam must keep two caches space!
//        // m = beta1*m + (1-beta1)*dx
//        // v = beta2*v + (1-beta2)*(dx**2)
//        // x += - learning_rate * m / (np.sqrt(v) + eps)
//    }
//};

} // namespace SuperNeurons

#endif //SUPERNEURONS_SOLVER_H
