#include <superneurons.h>
#include "testing.h"
#include <util/preprocess.h>
#include <util/print_util.h>

using namespace SuperNeurons;

class tensor_test : public TestSuite {
    
    public:
    
    void setup() {
    }
    
    void teardown() {
    }
    
};

ADDTEST(tensor_test, test_fft) {
    std::vector<tensor_t<float>* > *reg = new std::vector<tensor_t<float>* >();
    tensor_t<float>* t = new tensor_t<float>(1, 1, 5, 5, reg, GRAD, 1);
    
    t->init( new random_initializer_t<float>() );
    t->printTensorNoDebug("test fft=>input data");
    t->forward_fft();
    t->backward_fft();
    t->printTensorNoDebug("test fft=>reconstrcuted data");
    
    free(t);
    free(reg);
}
