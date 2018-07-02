//
// Created by ay27 on 8/15/17.
//

#include <superneurons.h>
using namespace std;
using namespace SuperNeurons;

int main(int argc, char** argv) {

    const int T = 100;

    const size_t MB = 1024*1024;

    double ts;
    double t1 = 0, t2 = 0;
    size_t size;

    //------------------------------------------------
    for (size_t size = 128; size <= 5*1024; size += 128) {
        t1 = 0;
        t2 = 0;
        float *ptr;
        for (int i = 0; i < T; ++i) {
            ts = get_cur_time();
            cudaMalloc(&ptr, size * MB);
            t1 += get_cur_time() - ts;

            ts = get_cur_time();
            cudaFree(ptr);
            t2 += get_cur_time() - ts;
        }
        t1 = t1 / (double)T;
        t2 = t2 / (double)T;

        printf("- %f\n", t1);

        printf("+ %f\n", t2);
    }
}
