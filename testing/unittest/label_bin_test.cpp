//
// Created by ay27 on 17/4/11.
//

#include <superneurons.h>
#include "testing.h"
#include <fstream>
#include <util/image_reader.h>

using namespace std;
using namespace SuperNeurons;

class test_label_bin : public TestSuite {
public:
    virtual void setup() {
        TestSuite::setup();
    }

    virtual void teardown() {
        TestSuite::teardown();
    }
};

ADDTEST(test_label_bin, read_bin) {
    ifstream bin("/Users/ay27/ClionProjects/superneurons/data/ilsvrc2012/label.bin");
    char meta[32];
    bin.read(meta, 32);
    uint64_t N = *((uint64_t *) meta);
    uint64_t C = *((uint64_t *) (meta + 8));
    uint64_t H = *((uint64_t *) (meta + 16));
    uint64_t W = *((uint64_t *) (meta + 24));
    printf("N=%d,C=%d,H=%d,W=%d\n", N, C, H, W);

    label_t *label = new label_t[N];
    bin.read((char *) label, N* sizeof(label_t));
    for (int i = 0; i < N; ++i) {
        printf("%u ", label[i]);
    }
    printf("\n");

    ifstream groundt("/Users/ay27/ClionProjects/superneurons/data/ilsvrc2012/val.txt");
    string line;
    vector<label_t > g;
    size_t pos;
    while (getline(groundt, line)) {
        pos = line.find_last_of(' ');
        g.push_back(atoi(line.substr(pos+1).c_str()));
    }

    for (int i = 0; i < g.size(); ++i) {
        printf("%u ", g[i]);
    }
    printf("\n");

//    arr_eq(label, g.data(), N);

}