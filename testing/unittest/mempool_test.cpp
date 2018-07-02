////
//// Created by ay27 on 7/24/17.
////
//
//#include <util/mempool.h>
//#include "testing.h"
//
//using namespace std;
//using namespace SuperNeurons;
//
//class test_mem_pool : public TestSuite {
//public:
//    virtual void setup() {
//        TestSuite::setup();
//    }
//
//    virtual void teardown() {
//        TestSuite::teardown();
//    }
//};
//
//
//ADDTEST(test_mem_pool, test_it) {
//    item_pool_t pool(2, 16, true);
//    float *x1, *x2, *x3;
//    printf("alloc\n");
//    expectT(pool.alloc((void**)&x1));
//    expectT(pool.alloc((void**)&x2));
//    expectF(pool.alloc((void**)&x3));
//
//    printf("fetch\n");
//    expectT(pool.fetch_item((void**)&x1));
//    expectT(pool.fetch_item((void**)&x2));
//    expectF(pool.fetch_item((void**)&x3));
//
//}