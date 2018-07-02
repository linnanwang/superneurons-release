//
// Created by ay27 on 17/4/3.
//

#include "testing.h"

class BasicTest : public TestSuite {
protected:
    int a, b;
public:

    void setup() {
        a = 1234;
        b = 4321;
    }

    void teardown() {
    }

};

ADDTEST(BasicTest, test_neq) {
    neq(a, b);
};

ADDTEST(BasicTest, test_eq) {
    a = b = 10;
    eq(a, b);
}

ADDTEST(BasicTest, test_arr) {
    char *x = (char *) "1234";
    char *y = (char *) "1234";
    arr_eq(x, y, 4);
    arr_neq(x, y, 4);
}

ADDTEST(BasicTest, test_almost) {
    float a = 1.0, b = 1.000002;
    almost_eq(a, b);
    almost_eq_precision(a, b, 0.0000001);
}

ADDTEST(BasicTest, test_arr_almost) {
    float x[4] = {1.0, 1.1, 2.0, 2.1};
    float y[4] = {1.0, 1.1, 2.0, 2.1};
    arr_almost_eq(x, y, 4);
    arr_almost_eq_precision(x, y, 4, 0.000001);
}

ADDTEST(BasicTest, test_expect) {
    expectT(1 == 1);
    expectF(1 != 1);
}

ADDTEST(BasicTest, with_except) {
    with_exception_start() {
        throw 10;
    } with_exception_end(int);
}
