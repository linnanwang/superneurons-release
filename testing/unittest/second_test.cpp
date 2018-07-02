//
// Created by ay27 on 17/4/3.
//

#include "testing.h"

class SecondTest : public TestSuite {
public:
    virtual void setup() {
        printf("setup");
    }

    virtual void teardown() {
        printf("teardown");
    }
};

ADDTEST(SecondTest, test_hello) {
    notNull("haha");
}