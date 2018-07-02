#if !defined(_NETWORK_H_)
#define _NETWORK_H_

#include <deque>
#include <registry.h>
#include <mem_control.h>
#include <util/common.h>
#include <util/error_util.h>
#include <layer/base_layer.h>
#include <layer/data_layer.h>
#include <util/superneurons_math.h>
#include <solver.h>
#include <util/error_util.h>
#include <stream_singleton.h>
#include <gpu_malloc.h>
#include <util/mem_util.h>
#include <layer/cudnn_convolution_layer.h>
#include <util/saver.h>
#include <string>

namespace SuperNeurons{

template <class value_type>
class network_t
{
private:
    /*-solver configurations-*/
    base_solver_t<value_type>* solver;
    const value_type clip_gradient_limit;
    size_t test_iter;
    size_t test_interval;


    /*-network configurations-*/
    size_t GPU_id;
    bool is_forward_setup;
    bool is_testing_ready;
    bool is_backward_setup;
    bool is_network_computable;
    math_util<value_type> math;
    cudnnHandle_t   cudnn_handle;
    cublasHandle_t  cublas_handle;
    cudnnDataType_t cudnn_data_type;
    cudaStream_t stream = stream_singleton::get_compute_stream();


    /*-computation route-*/
    //we maintain the uni-direction of forward_backward route,
    //this guides the creation, offloadig and deletion of tensors
    //std::vector<std::pair<int, net_comp> >      net_comp_route;
    //std::map<int, std::vector< tensor_t<value_type>* > > tensor_by_layer;

    /*--test swapping--*/
    //we do test by swapping the head of data reader,
    //softmax_loss_layer is tracked for backward computation
    //train
    base_layer_t<value_type>*  train_data_layer       = NULL;
    base_layer_t<value_type>*  softmax_loss_layer     = NULL;
    //test
    base_layer_t<value_type>*  test_data_layer        = NULL;
    /*--registry records the info of every tensor--*/
    registry_t<value_type> *reg;
    mem_controller_t<value_type> mem_controller;

    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnn_handle) );
        cudnnSetStream(cudnn_handle, stream);
        checkCublasErrors( cublasCreate(&cublas_handle) );
        cublasSetStream(cublas_handle, stream);
    }

    void destroyHandles()
    {
        checkCUDNN( cudnnDestroy(cudnn_handle) );
        checkCublasErrors( cublasDestroy(cublas_handle) );
    }

    void gradient_check_kernel(int l_id, size_t n, size_t c, size_t h, size_t w, tensor_t<value_type>* data, tensor_t<value_type>* diff, const char* str);

    void update_kernel( base_layer_t<value_type>* l, size_t iter);

    void regularization(base_layer_t<value_type>* l);

    void calculate_update_value(base_layer_t<value_type>* l);

    void forward_test(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* acc);

    void forward_kernel(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss);

    void backward_with_update_kernel(base_layer_t<value_type>* l, size_t iter);

    void backward_kernel(base_layer_t<value_type>* b);

    void fsetup_kernel(base_layer_t<value_type>* start_layer);

    void bsetup_kernel(base_layer_t<value_type>* start_layer);

    void test();

    void write_tensor(int32_t layer_id, tensor_t<value_type> *t, std::ofstream *out);
    void read_tensor(int32_t *layer_id, tensor_t<value_type> **t, std::ifstream *in);
    
    void meta_setup() {
        //CAUTION: the sequence of registry and mem_controller matters
        printf("************network layer configurations************\n");
        reg->print_net_comp_route();
        reg->print_net_layers();
        
        reg->register_tensors_by_layers();
        reg->print_tensors_by_layers();
        printf("****************************************************\n");
        
        printf("*****************network dependencies***************\n");
        printf("---------------------FORWARD-------------------------\n");
        reg->print_forward_dependency();
        printf("---------------------BACKWARD------------------------\n");
        reg->print_backward_dependency();
        printf("****************************************************\n");
        
        printf("*****************init memory control****************\n");
        mem_controller.init(reg);
        mem_controller.print_regulated_tensors();
        printf("****************************************************\n");
    }

    std::vector<double> network_perf_profile();

    std::shared_ptr<std::thread> query_thread;
    std::atomic_bool query_stop;

    void init_gpu_mem_query() {
        query_stop = false;
        query_thread = std::make_shared<std::thread>([&]() {

#ifdef __linux__
            // must wait a little to set affinity
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                std::cout << "gpu mem query thread run on CPU " << sched_getcpu() << "\n";
#endif

            size_t max_usage = 0;
            size_t tmp;
            double ts = get_cur_time();
            while ( ! query_stop.load() ) {
                tmp = query_used_mem();
                if (tmp > max_usage) {
                    max_usage = tmp;
                }
                if (get_cur_time() - ts > 100.0) {
                    printf("QUERY====>max gpu usage : %f MB\n", BYTE_TO_MB(max_usage));
                    ts = get_cur_time();
                    max_usage = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });

        set_cpu_affinity(query_thread->native_handle(), -2);
    }

public:

    network_t(base_solver_t<value_type>* _solver):is_network_computable(false), solver(_solver), clip_gradient_limit(35.0), test_iter(0)
    {
        google::InitGoogleLogging("");
        FLAGS_logtostderr = 1;

        reg = new registry_t<value_type>();

#ifdef LIVENESS
        printf("LIVENESS !!!!!\n");
#endif
#ifdef RECOMPUTE_ON
        printf("RECOMPUTE_ON !!!!!!!!\n");
#endif
#ifdef LARGER
        printf("LARGER !!!!\n");
#endif
#ifdef LRU_ON
        printf("LRU_ON !!!\n");
#endif
#ifdef BLASX_MALLOC
        printf("BLASX_MALLOC !!!!\n");
#endif
        //set affinity
        set_main_thread_cpu_affinity(1);

        init_gpu_mem_query();


        is_forward_setup  = false;
        is_backward_setup = false;
        is_testing_ready  = false;

        switch (sizeof(value_type))
        {
            case 2: cudnn_data_type = CUDNN_DATA_HALF;   break;
            case 4: cudnn_data_type = CUDNN_DATA_FLOAT;  break;
            case 8: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        createHandles();
    };

    ~network_t()
    {
        // we first destory data layer because the ParallelReader must be destoryed before registry
        delete train_data_layer;
        delete test_data_layer;

        delete reg;

        query_stop = true;
        query_thread->join();
        //the sequence matters
        destroyHandles();
        //all tensors will be deleted in the registry class

        // finish all computation, destroy compute stream
        stream_singleton::destory_stream();

        // destroy global blasx_malloc_t
        blasx_gpu_singleton::destroy_all_instance();
    }

    math_util<value_type>* get_math_util() {
        return &(this->math);
    }

    cudnnHandle_t* get_cudnn_handle() {
        return &(this->cudnn_handle);
    }

    cublasHandle_t* get_cublas_handle() {
        return &(this->cublas_handle);
    }

    registry_t<value_type>* get_registry() {
        return this->reg;
    }


    void fsetup(base_layer_t<value_type>* start_layer) {
        if(this->train_data_layer == NULL) {
            this->train_data_layer = start_layer;
        } else {
            printf("fsetup train data layer could only be set once!! line 12@network.cpp\n");
            exit(1);
        }
        fsetup_kernel(start_layer);
        this->is_forward_setup = true;
    }

    void bsetup(base_layer_t<value_type>* end_layer) {
        if(this->softmax_loss_layer == NULL) {
            this->softmax_loss_layer = end_layer;
        } else {
            printf("bsetup softmax layer could only be set once!! line 43@network.cpp\n");
            exit(1);
        }
        bsetup_kernel(end_layer);
        this->is_backward_setup = true;
        meta_setup();
    }

    void setup_test(base_layer_t<value_type>* test_data_layer,
                    size_t iter);

    value_type forward(network_stage stage) {
        assert(this->train_data_layer != NULL);
        base_layer_t<value_type>* n = this->train_data_layer;
        std::vector<value_type> loss;
        forward_kernel(stage, n, &loss);
        return loss[0];
    }

    void backward() {
        assert(this->softmax_loss_layer != NULL);
        base_layer_t<value_type>* n = this->softmax_loss_layer;
        backward_kernel(n);
    }

    void backward_with_update(size_t iter) {
        assert(this->softmax_loss_layer != NULL);

        base_layer_t<value_type>* n = this->softmax_loss_layer;
        backward_with_update_kernel(n, iter);
    }
    
    void update( size_t iter) {
        assert(this->train_data_layer != NULL);
        base_layer_t<value_type>* start = this->train_data_layer;
        update_kernel(start, iter);
    }

    void train(size_t iter, size_t tracking_window, size_t test_interval, network_saver *saver=NULL) {
        assert(is_forward_setup == true);
        assert(is_testing_ready == true);
        assert(is_backward_setup == true);

        size_t curt_mem = query_used_mem();
        printf("after setup the memory used:%f\n", BYTE_TO_MB(curt_mem));

        value_type loss = 0;
        value_type running_std     = 0;
        value_type running_average = 0;
        value_type threshold       = 0;
        std::deque<value_type> loss_queue;
        double speed_start = get_cur_time();


        for(size_t i = 1; i <= iter; i++) {

            if (i % 50000 == 0 && saver != NULL) {
                saver->save();
            }

            double iter_start = get_cur_time();
            /*--network calculation--*/
            loss = forward(NET_TRAIN);
            backward();
            update(i);
            //backward_with_update(i);
            /*----loss statistics----*/
            if(loss_queue.size() < tracking_window) {
                if (std::isfinite(loss)) {
                    loss_queue.push_back(loss);
                    running_average = ((i-1)*running_average+loss)/(i);
                }
            } else {
                value_type loss_to_go = loss_queue.front();
                running_average = (running_average*tracking_window - loss_to_go + loss)/tracking_window;
                loss_queue.pop_front();
                loss_queue.push_back(loss);
            }
            running_std = 0;
            for(unsigned i = 0; i < loss_queue.size(); i++) {
                running_std += ((loss_queue[i] - running_average)*(loss_queue[i] - running_average));
            }
            running_std        = sqrt(running_std / (value_type) loss_queue.size());
            threshold = running_average + 3*running_std;

            if (i % 20 == 0) {
                double speed_end   = get_cur_time();
                double speed_time  = speed_end - speed_start;
                size_t batch_size  = ((data_layer_t<value_type>*) train_data_layer)->get_batch_size();
                double train_imgs  = batch_size*20.0f;
                double train_speed = train_imgs / speed_time;
                speed_start = get_cur_time();
                time_t tt = time(NULL);
                tm* t= localtime(&tt);
                printf("%d-%02d-%02d %02d:%02d:%02d-----iter:%zu--lr:%.10f--loss:%f--avg:%f--std:%f--threshold:%f--%f:img/s\n",
                       t->tm_year + 1900,
                       t->tm_mon + 1,
                       t->tm_mday,
                       t->tm_hour,
                       t->tm_min,
                       t->tm_sec,
                       i, solver->get_lr(), loss, running_average, running_std, threshold, train_speed);
            }
            double iter_end = get_cur_time();
            if (i < 10) {
                fprintf(stderr,"-----iter:%zu--lr:%f--loss:%f--avg:%f--std:%f--threshold:%f--iter time:%f\n", i, solver->get_lr(), loss, running_average, running_std, threshold, iter_end - iter_start);
            }
            if(i % test_interval == 0) {
                test();
            }

            solver->update_lr(i, running_average);
        }
    }

    void gradient_check(int layer_id);
};

}
#endif // _NETWORK_H_
