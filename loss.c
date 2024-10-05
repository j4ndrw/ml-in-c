#include "tensor.h"
#include "variable.h"

Variable loss_mse(Variable *actual, Variable *predicted) {
    var_from(sum, tensor_new_scalar(0));
    var_from(pow, tensor_new_scalar(2));
    var_from(n, tensor_new_scalar(actual->items.length));

    var_expr(diff, op(actual, -, predicted));
    var_expr(squared, op(&diff, <^>, &pow));
    var_expr(acc_sum, op(&sum, [+], &squared));
    var_expr(result, op(&acc_sum, /, &n));
    return result;
}
