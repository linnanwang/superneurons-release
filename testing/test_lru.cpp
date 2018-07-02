//
// Created by ay27 on 8/18/17.
//

#include <util/lru.h>
#include <superneurons.h>

using namespace std;
using namespace SuperNeurons;

int main(int argc, char **argv) {
    lru_list_t lru;
    std::vector<tensor_t<float> *> *reg = new std::vector<tensor_t<float> *>();
    tensor_t<float> *t1 = new tensor_t<float>(2, 3, 4, 5, reg, DATA, 0);
    tensor_t<float> *t2 = new tensor_t<float>(2, 3, 4, 5, reg, DATA, 0);
    tensor_t<float> *t3 = new tensor_t<float>(2, 3, 4, 5, reg, DATA, 0);
    tensor_t<float> *t4 = new tensor_t<float>(2, 3, 4, 5, reg, DATA, 0);

    lru.update(t1);
    lru.update(t2);
    lru.update(t3);
    lru.update(t4);
    lru.print_list();

    lru.update(t3);
    lru.print_list();

    lru.update(t3);
    lru.print_list();

    lru.remove_oldest();
    lru.print_list();
    printf("after remove oldest: %p %p %p %p\n", t1, t2, t3, t4);

    lru.remove_item(lru.get_item(-2));
    lru.print_list();
    printf("after remove oldest: %p %p %p %p\n", t1, t2, t3, t4);

    lru.update(t1);
    lru.update(t4);
    lru.print_list();
    lru.remove_oldest();
    lru.remove_oldest();
    lru.remove_oldest();
    lru.remove_oldest();
    lru.remove_oldest();
    lru.remove_oldest();
    lru.remove_item(lru.get_item(1));
    lru.remove_item(lru.get_item(-1));

    lru.print_list();
}
