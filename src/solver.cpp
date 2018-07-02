//
// Created by ay27 on 7/18/17.
//

#include <solver.h>


namespace SuperNeurons {

template<class value_type>
void base_solver_t<value_type>::set_lr_decay_policy(lr_decay_type _decay_type,
                                                    std::initializer_list<double> points,
                                                    std::initializer_list<value_type> lrs) {
    if (points.size() != lrs.size()) {
        fprintf(stderr, "the size of iters and lrs should be equal when setting the lr decay");
        exit(-1);
    }

    this->decay_type = _decay_type;

    lr_policy.clear();
    typename std::initializer_list<double>::iterator it1 = points.begin();
    typename std::initializer_list<value_type>::iterator it2 = lrs.begin();
    for (size_t i = 0; i < points.size(); ++i, ++it1, ++it2) {
        lr_policy.push_back(std::make_pair(*it1, *it2));
    }
}

template<class value_type>
void base_solver_t<value_type>::update_lr(size_t _iter, double avg_loss) {
    if (lr_policy.empty()) {
        return;
    }

    for (size_t i = start_idx; i < lr_policy.size(); ++i) {
        if (this->decay_type == ITER) {
            if (fabs(lr_policy[i].first - _iter) < 0.01) {
                this->lr = lr_policy[i].second;
                start_idx = i;
                break;
            }
        } else if (this->decay_type == LOSS) {
            if (lr_policy[i].first - avg_loss <= 0.001) {
                this->lr = lr_policy[i].second;
                start_idx = i;
                break;
            }
        }
    }
}

template<class value_type>
void base_solver_t<value_type>::regularization(cublasHandle_t *handle,
                                               tensor_t<value_type> *p,
                                               tensor_t<value_type> *p_grad) {
    //TODO, will add additional L1 regularization
    const value_type wd = weight_decay;
    const size_t total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();

#ifdef DEBUG_SOLVER
    printf("----------------Regularization:lr%f----------------\n", wd);
    p->printTensor("weight before adding weight_grad");
    p_grad->printTensor("weight grad");
#endif

    //weight decay
    //w_i+1 = w_i - lr*(dEdwi + weigth_decay*w_i)
    cublas_axpy(handle,
                total_params,
                &(wd),
                p->get_gpu_ptr(), 1,
                p_grad->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p->printTensor("weight after adding weight_grad");
    p_grad->printTensor("weight grad after adding weight_grad");
    printf("----------------Regularization END----------------\n");
#endif

}

template<class value_type>
void base_solver_t<value_type>::update(cublasHandle_t *handle, size_t _iter, tensor_t<value_type> *weight,
                                       tensor_t<value_type> *bias,
                                       tensor_t<value_type> *weight_grad, tensor_t<value_type> *bias_grad,
                                       tensor_t<value_type> *weight_prev, tensor_t<value_type> *bias_prev,
                                       bool enable_bias) {
    if (weight == NULL || bias == NULL) {
        return;
    }

    // we update lr according to the lr_policy here, then do the concrete update in update_kernel
//    update_lr(_iter);

    // TODO : does it correct to do the regularization **before** momentum or other solver ?
    // TODO : does it should do the regularization to bias term ?
    regularization(handle, weight, weight_grad);
//    if (enable_bias) {
//        regularization(handle, bias, bias_grad);
//    }
    update_kernel(handle, weight, bias, weight_grad, bias_grad, weight_prev, bias_prev, enable_bias);
}

/*--------------------------------------------------------------------------------------------------------------------*/

template<class value_type>
void momentum_solver_t<value_type>::calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p,
                                                     tensor_t<value_type> *p_grad,
                                                     tensor_t<value_type> *p_prev,
                                                     bool is_bias) {
    const value_type one = 1.0f;
    value_type local_lr = this->lr;
    if(is_bias) {
        local_lr = local_lr*2;
    }
    int total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();
    
    momentum_update( total_params, p_grad->get_gpu_ptr(), p_prev->get_gpu_ptr(), m, local_lr );
    const value_type none = one * -1;
    cublas_axpy(handle, total_params, &none, p_grad->get_gpu_ptr(), 1, p->get_gpu_ptr(), 1);
}

/*--------------------------------------------------------------------------------------------------------------------*/

template<class value_type>
void sgd_solver_t<value_type>::calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p,
                                                tensor_t<value_type> *p_grad) {
    size_t total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();
    value_type local_lr = this->lr * -1;
    value_type one = 1;

#ifdef DEBUG_SOLVER
    printf("----------------Cal Update Value:lr%f----------------\n", this->get_lr());
    p->printTensor("weight");
    p_grad->printTensor("weight_grad");
#endif

    // x = x - lr*dx
    cublas_scal(handle, total_params, &local_lr, p_grad->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p_grad->printTensor("weight_grad = -lr * weight_grad");
#endif

    cublas_axpy(handle, total_params, &one, p_grad->get_gpu_ptr(), 1, p->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p->printTensor("weight = weight - lr*weight_grad");
    printf("----------------Cal Update Value End----------------\n");
#endif
}

/*--------------------------------------------------------------------------------------------------------------------*/

template<class value_type>
void nesterov_solver_t<value_type>::calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p,
                                                     tensor_t<value_type> *p_grad,
                                                     tensor_t<value_type> *p_prev) {
    value_type local_lr = this->lr * -1;
    size_t total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();
    value_type one = 1.0;

#ifdef DEBUG_SOLVER
    printf("----------------Cal Update Value:lr%f----------------\n", this->get_lr());
    p->printTensor("weight");
    p_grad->printTensor("weight_grad");
    p_prev->printTensor("weight_prev");
#endif

    // v = m*v_prev - lr*dx
    cublas_scal(handle, total_params, &local_lr, p_grad->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p_grad->printTensor("weight_grad = -lr * weight_grad");
#endif

    cublas_axpy(handle, total_params, &m, p_prev->get_gpu_ptr(), 1, p_grad->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p_grad->printTensor("weight_grad = m*v_prev - lr*weight_grad");
#endif

    value_type m1 = 1.0 + m;
    value_type _m = -m;

    // x += -m*v_prev + (1+m)*v
    // the immediate result saved in p_prev
    cublas_scal(handle, total_params, &_m, p_prev->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p_prev->printTensor("weight_prev = -m * weight_prev");
#endif

    cublas_axpy(handle, total_params, &m1, p_grad->get_gpu_ptr(), 1, p_prev->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p_prev->printTensor("weight_prev = -m*weight_prev + (1+m)*weight_grad");
#endif

    cublas_axpy(handle, total_params, &one, p_prev->get_gpu_ptr(), 1, p->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p->printTensor("x += -m*weight_prev+(1+m)*weight_grad");
#endif

    //copy the curt update value to the prev_update
    checkCudaErrors(
            cudaMemcpy((void *) p_prev->get_gpu_ptr(),
                       (void *) p_grad->get_gpu_ptr(),
                       sizeof(value_type) * total_params, cudaMemcpyDeviceToDevice));

#ifdef DEBUG_SOLVER
    printf("----------------Cal Update Value End----------------\n");
#endif
}


template<class value_type>
void adagrad_solver_t<value_type>::calculate_update(cublasHandle_t *handle, tensor_t<value_type> *p,
                                                    tensor_t<value_type> *p_grad,
                                                    tensor_t<value_type> *p_cache) {
    size_t total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();

#ifdef DEBUG_SOLVER
    printf("----------------Cal Update Value:lr%f----------------\n", this->get_lr());
    p->printTensor("weight");
    p_grad->printTensor("weight_grad");
    p_cache->printTensor("weight cache");
#endif

    // this will calculate :
    // p_cache += dx**2
    // dx = learning_rate * dx / (np.sqrt(cache) + eps)
    adagrad_update(total_params, p_grad->get_gpu_ptr(), p_cache->get_gpu_ptr(), eps, this->lr);

#ifdef DEBUG_SOLVER
    p_cache->printTensor("p_cache += weight_grad**2");
    p_grad->printTensor("weight_grad = learning_rate * weight_grad / (np.sqrt(cache) + eps)");
#endif

    // cache += dx**2
    // x += - learning_rate * dx / (np.sqrt(cache) + eps)
    value_type one = -1.0f;
    cublas_axpy(handle, total_params, &one, p_grad->get_gpu_ptr(), 1, p->get_gpu_ptr(), 1);
#ifdef DEBUG_SOLVER
    p->printTensor("weight += -dx");
    printf("----------------Cal Update Value End----------------\n");
#endif
}


template<class value_type>
void rmsprop_solver_t<value_type>::calculate_update(cublasHandle_t *handle,
                                                    tensor_t<value_type> *p,
                                                    tensor_t<value_type> *p_grad,
                                                    tensor_t<value_type> *p_cache) {
    size_t total_params = p->get_N() * p->get_C() * p->get_H() * p->get_W();

#ifdef DEBUG_SOLVER
    printf("----------------Cal Update Value:lr%f----------------\n", this->get_lr());
    p->printTensor("weight");
    p_grad->printTensor("weight_grad");
    p_cache->printTensor("weight cache");
#endif

    // cache = decay_rate * cache + (1 - decay_rate) * dx**2
    // dx = learning_rate * dx / (np.sqrt(cache) + eps)

    rmsprop_update(total_params, p_grad->get_gpu_ptr(), p_cache->get_gpu_ptr(), rms_decay, eps, this->lr);

#ifdef DEBUG_SOLVER
    p_cache->printTensor("cache = decay_rate * cache + (1 - decay_rate) * weight_grad**2");
    p_grad->printTensor("weight_grad = learning_rate * weight_grad / (np.sqrt(cache) + eps)");
#endif

    value_type one = -1.0f;
    cublas_axpy(handle, total_params, &one, p_grad->get_gpu_ptr(), 1, p->get_gpu_ptr(), 1);

#ifdef DEBUG_SOLVER
    p->printTensor("weight += -weight_grad");
    printf("----------------Cal Update Value End----------------\n");
#endif
}


INSTANTIATE_CLASS(base_solver_t);

INSTANTIATE_CLASS(sgd_solver_t);

INSTANTIATE_CLASS(momentum_solver_t);

INSTANTIATE_CLASS(nesterov_solver_t);

INSTANTIATE_CLASS(adagrad_solver_t);

INSTANTIATE_CLASS(rmsprop_solver_t);

} // namespace SuperNeurons
