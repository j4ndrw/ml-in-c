#include "tensor.h"
#include "variable.h"

Variable loss_mse(Variable *actual, Variable *predicted) {
    assert(actual->items.length > 0 &&
           "Tensors need to have a length > 0 to calc mean squared error");
    assert(predicted->items.length > 0 &&
           "Tensors need to have a length > 0 to calc mean squared error");

    var_from(sum, tensor_new_scalar(0));
    var_from(pow, tensor_new_scalar(2));
    var_from(n, tensor_new_scalar(actual->items.length));

    var_expr(diff, op(actual, -, predicted));
    var_expr(squared, op(&diff, <^>, &pow));
    var_expr(acc_sum, op(&sum, [+], &squared));
    return op(&acc_sum, /, &n);
}
