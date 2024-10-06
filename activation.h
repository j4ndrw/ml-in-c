#include "tensor.h"
#include "variable.h"
#include <math.h>

#define activation_relu(result, x)                                             \
    var_from(relu_half, tensor_new_scalar(0.5));                               \
    var_from(relu_neg_one, tensor_new_scalar(-1));                             \
    var_expr(relu_neg_x, op(&x, <*>, &relu_neg_one));                          \
    var_expr(relu_sum, op(&x, +, &relu_neg_x));                                \
    var_expr(result, op(&relu_sum, <*>, &relu_half))

#define activation_sigmoid(result, x)                                          \
    Variable result;                                                           \
    do {                                                                       \
        var_from(sig_one, tensor_new_scalar(1));                               \
        var_from(sig_neg_one, tensor_new_scalar(-1));                          \
        var_from(sig_neg_one_2, tensor_new_scalar(-1));                        \
        var_expr(sig_neg_x, op(&x, <*>, sig_neg_one));                         \
        var_expr(sig_exp_inv, op_self(&sig_neg_x, exp));                       \
        var_expr(sig_exp_inv_plus_one, op(&sig_exp_inv, <+>, &sig_one));       \
        result = op(&sig_exp_inv_plus_one, <^>, sig_neg_one_2);                \
    } while (0)
