////
//// Created by ay27 on 17/4/8.
////
//
//#include "testing.h"
//#include "../../tools/compute_image_mean.cpp"
//
//class compute_image_mean_test: public TestSuite {
//public:
//    virtual void setup() {
//        printf("\n");
//    }
//
//    virtual void teardown() {
//        printf("\n");
//    }
//};
//
//ADDTEST(compute_image_mean_test, test_normal) {
//    const char* input_binary_path = "/tmp/input";
//    const char* output_mean_path = "/tmp/output";
//
//    ofstream output(input_binary_path, ios::out | ios::binary);
//
//    size_t N = 100, C = 3, H = 10, W = 10;
//    uint8_t* image_data = new uint8_t[C*H*W];
//    for (int i = 0; i < C * H * W; ++i) {
//        image_data[i] = 10;
//    }
//
//}