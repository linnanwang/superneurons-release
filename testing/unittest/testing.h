//
// Created by ay27 on 17/4/3.
//

#ifndef PROJECT_TESTING_H
#define PROJECT_TESTING_H

#include <vector>
#include <stdio.h>
#include <exception>
#include <math.h>

/////////////////////////////////////////////////////////////////////////////////
// print color

#define NONE                "\e[0m"
#define L_RED               "\e[1;31m"
#define L_GREEN             "\e[1;32m"
#define BOLD                "\e[1m"
#define L_BLUE              "\e[1;34m"

/////////////////////////////////////////////////////////////////////////////////

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#define __join(a, b) a#b

/////////////////////////////////////////////////////////////////////////////////
#define error(msg) { \
    addMsg(L_RED "  Error test \"" msg "\" failed\n");\
    __set_err_flag__();\
}

#define success(msg) addMsg(L_GREEN "  Ok test \"" msg "\" success\n")

/////////////////////////////////////////////////////////////////////////////////
// basic testing

#define eq(a, b) \
    if ((a) == (b)) success("equal"); \
    else error("test equal failed")

#define neq(a, b) if ((a) != (b)) success("not equal"); \
    else error("test not equal failed")

#define expectT(condition) if (condition) success("expect true"); \
    else error("expect true")

#define expectF(condition) if (!(condition)) success("expect false"); \
    else error("expect false")

#define notNull(obj) if ((obj) != NULL) success("no zero"); \
    else error("no zero")

/////////////////////////////////////////////////////////////////////////////////
// almost precision testing

#define almost_eq_precision(a, b, eps) \
    if (fabs((a)-(b))<eps) success(__join("almost equal with precision ", eps));\
    else error(__join("almost equal with precision ", eps))

#define almost_eq(a, b) \
    if (fabs((a)-(b))<0.0001) success("almost equal with precision 0.0001");\
    else error("almost equal with precision 0.0001")

/////////////////////////////////////////////////////////////////////////////////
// array testing

#define arr_eq(arrA, arrB, len) \
    bool __arrA##arrB_eq__ = true; \
    for (size_t i=0; i<len; i++) { \
        if (arrA[i] != arrB[i]) { __arrA##arrB_eq__ = false; break;} \
    } \
    if (__arrA##arrB_eq__) success("array equal"); else error("array equal")

#define arr_neq(arrA, arrB, len) \
    bool __arrA##arrB_neq__ = true; \
    for (size_t i=0; i<len; i++) { \
        if (arrA[i] != arrB[i]) { __arrA##arrB_neq__ = false; break;} \
    } \
    if (!__arrA##arrB_neq__) success("array not equal"); else error("array not equal")

#define arr_almost_eq(arrA, arrB, len) \
    bool __arrA##arrB_almost_eq__ = true; \
    for (size_t i=0; i<len; i++) { \
        if (fabs(arrA[i] - arrB[i])>0.0001) { __arrA##arrB_almost_eq__ = false; break;} \
    } \
    if (__arrA##arrB_almost_eq__) success("array almost equal with precision 0.0001");\
    else error("array almost equal with precision 0.0001")

#define arr_almost_eq_precision(arrA, arrB, len, precision) \
    bool __arrA##arrB_almost_eq_precision__ = true; \
    for (size_t i=0; i<len; i++) { \
        if (fabs(arrA[i] - arrB[i])>precision) { __arrA##arrB_almost_eq_precision__ = false; break;} \
    } \
    if (__arrA##arrB_almost_eq_precision__) success(__join("array almost equal with precision ", precision));\
    else error(__join("array almost equal with precision ", precision))

/////////////////////////////////////////////////////////////////////////////////
// exception testing

#define with_exception_start() try

#define with_exception_end(e) \
    catch (e err) { \
        success(__join("with exception ", e));} \
    catch(...) { \
        error(__join("with exception ", e)); \
    }

/////////////////////////////////////////////////////////////////////////////////
// ADDTEST macro

#define GENERATE_CLASS_NAME(cls, func) \
    cls##func

#define ADDTEST(parent_class, func_name) \
class GENERATE_CLASS_NAME(parent_class, func_name) : public parent_class { \
public: \
    GENERATE_CLASS_NAME(parent_class, func_name)() {} \
    void TestBody();\
    void before() { \
        err_flag=true; \
        printf(NONE "case %d \t%s::%s ... ", Test::get_instance()->get_invoke_cnt(), #parent_class, #func_name); \
    } \
    void after() { \
        if (err_flag) printf(L_GREEN "success\n");\
        else { \
            printf(L_RED "error\n");\
            for (size_t i=0; i<msgs.size(); i++){ \
                printf("%s",msgs[i]); \
            }\
        } \
        Test::get_instance()->tick(); \
    }\
private: \
    int invoke_cnt = 1; \
    static const unsigned long __a_hack_trick__; \
    DISABLE_COPY_AND_ASSIGN(GENERATE_CLASS_NAME(parent_class, func_name)); \
    std::vector<const char*> msgs; \
    void addMsg(const char* msg) {msgs.push_back(msg);} \
};\
const unsigned long GENERATE_CLASS_NAME(parent_class, func_name)::__a_hack_trick__ = \
    Test::get_instance()->add_test(new GENERATE_CLASS_NAME(parent_class, func_name)); \
void GENERATE_CLASS_NAME(parent_class, func_name)::TestBody()

/////////////////////////////////////////////////////////////////////////////////

class TestSuite {
public:
    virtual void setup() {}
    virtual void teardown() {}
    virtual void TestBody() = 0;
    virtual void before() = 0;
    virtual void after() = 0;

protected:
    void __set_err_flag__() {
        err_flag = false;
    }

    bool err_flag;
};

class Test {
private:
    Test() {}

    std::vector<TestSuite *> testCases;
    int invoke_cnt = 1;
public:
    static Test *get_instance() {
        static Test instance;
        return &instance;
    }

    void tick() {
        invoke_cnt ++;
    }

    int get_invoke_cnt() {
        return invoke_cnt;
    }

    void run() {
        printf("size of test cases: %lu\n", testCases.size());
        for (size_t i = 0; i < testCases.size(); ++i) {
            testCases[i]->before();
            testCases[i]->setup();
            testCases[i]->TestBody();
            testCases[i]->teardown();
            testCases[i]->after();
        }
    }

    unsigned long add_test(TestSuite *test_case) {
        testCases.push_back(test_case);
        return testCases.size();
    }

DISABLE_COPY_AND_ASSIGN(Test);
};

#endif //PROJECT_TESTING_H
