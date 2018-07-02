#if !defined(_MATH_H_)
#define _MATH_H_

#include <cmath>

namespace SuperNeurons{

template <class value_type>
class math_util {
public:
    value_type log_(value_type i) {
        return std::log(i);
    }
    
    value_type abs_(value_type i) {
        return std::abs(i);
    }
    
    value_type max_(value_type a, value_type b) {
        return std::max(a,b);
    }
    
};

} //SuperNeurons namespace

#endif
