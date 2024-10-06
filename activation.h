#include "tensor.h"
#include "variable.h"

#define activation_relu(result, x)                                             \
    var_from(relu_half, tensor_new_scalar(0.5));                               \
    var_from(relu_neg_one, tensor_new_scalar(-1));                             \
    var_expr(relu_neg_x, op(&x, <*>, &relu_neg_one));                            \
    var_expr(relu_sum, op(&x, +, &relu_neg_x));                                  \
    var_expr(result, op(&relu_sum, <*>, &relu_half))
