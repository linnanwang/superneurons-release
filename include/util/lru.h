//
// Created by ay27 on 8/18/17.
//

#ifndef SUPERNEURONS_LRU_H
#define SUPERNEURONS_LRU_H

#include <util/common.h>

namespace SuperNeurons {

class lru_block_t {
public:
    void* item;
    lru_block_t *prior;
    lru_block_t *next;

    lru_block_t(void* item, lru_block_t *prior, lru_block_t *next) : item(item), prior(prior), next(next) {}

    ~lru_block_t() {
        // we don't delete tensor here!!!!
        // and we don't delete prior and next!!!!
    }
};


class lru_list_t {
private:
    lru_block_t *head;
    lru_block_t *tail;

public:
    lru_list_t() {
        head = NULL;
        tail = NULL;
    }

    ~lru_list_t() {
        lru_block_t* tmp;
        while (head != NULL) {
            tmp = head;
            head = head->next;
            delete tmp;
        }
    }

    void update(void* item) {
        if (item == NULL) {
            return;
        }
        lru_block_t *it = find(item);
        if (it == NULL) {
            // insert

            if (head == NULL) {
                head = new lru_block_t(item, NULL, NULL);
                // set tail here
                tail = head;
            } else {
                lru_block_t *tmp = new lru_block_t(item, NULL, head);
                head->prior = tmp;
                head = tmp;
            }
            return;
        }

        if (it == head) {
            return;
        }

        if (it == tail) {
            tail = it->prior;
            it->prior->next = NULL;

            it->next = head;
            head->prior = it;
            head = it;
        } else if (it != head) {
            it->prior->next = it->next;
            it->next->prior = it->prior;

            it->next = head;
            head->prior = it;
            it->prior = NULL;

            head = it;
        }
    }

    void* remove_oldest() {
        if (tail == NULL) {
            return NULL;
        }
        void* res = tail->item;
        if (head != tail) {
            tail->prior->next = NULL;
            auto tmp = tail;
            tail = tail->prior;
            delete tmp;
        } else {
            delete tail;
            head = NULL;
            tail = NULL;
        }
        return res;
    }

    void* remove_item(lru_block_t *it) {
        void* res = NULL;
        if (it == NULL) {
            return NULL;
        }
        if (it == tail) {
            return remove_oldest();
        } else if (it == head) {
            auto tmp = head;
            head = head->next;
            head->prior = NULL;
            res = it->item;
            delete tmp;
        } else {
            it->prior->next = it->next;
            it->next->prior = it->prior;
            res = it->item;
            delete it;
        }
        return res;
    }

    lru_block_t *find(void* t) {
        if (t == NULL) {
            return NULL;
        }
        lru_block_t *it = head;
        while (it != NULL) {
            if (it->item == t) {
                return it;
            }
            it = it->next;
        }
        return NULL;
    }

    lru_block_t *get_item(int idx) {
        lru_block_t *res;
        if (idx >= 0) {
            res = head;
            for (int i = 0; i < idx; ++i) {
                if (res != NULL) {
                    res = res->next;
                } else {
                    fprintf(stderr, "out of index when get_item in lru, idx: %d !!!!\n", idx);
                    return NULL;
                }
            }
        } else {
            res = tail;
            // -1 == tail
            for (int i = 0; i < -idx-1; ++i) {
                if (res != NULL) {
                    res = res->prior;
                } else {
                    fprintf(stderr, "out of index when get_item in lru, idx: %d !!!!\n", idx);
                    return NULL;
                }
            }
        }
        return res;
    }

    void print_list() {
        printf("\n------- LRU List -------\n");
        printf("head: %p, tail: %p\n", head, tail);
        lru_block_t *it = head;
        while (it != NULL) {
            printf("block %p: item %p, prior: %p, next: %p\n", it, it->item, it->prior, it->next);
            it = it->next;
        }
        printf("\n");
    }
};


class lru_singleton {
private:
    lru_list_t* list;
    static lru_singleton* instance;

    explicit lru_singleton() {
        list = new lru_list_t();
    }

    ~lru_singleton() {
        delete list;
    }

public:

    static lru_list_t* get_lru() {
        if (instance == NULL) {
            instance = new lru_singleton();
        }
        return instance->list;
    }

    static void destory() {
        delete instance;
    }
};

} // namespace SuperNeurons

#endif //SUPERNEURONS_LRU_H
