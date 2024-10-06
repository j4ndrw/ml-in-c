#include "tensor.h"
#include "variable.h"

#define calc_avg_2(result, x, y)                                               \
    Variable result;                                                           \
    do {                                                                       \
        var_from(n, tensor_new_scalar(2));                                     \
        var_expr(sum, op(&x, +, &y));                                          \
        result = op(&sum, /, &n);                                              \
    } while (0)
