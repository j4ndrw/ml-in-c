#include "variable.h"

Variable loss_mse(Variable *a, Variable *b) {
    assert(a->items.length > 0 &&
           "Tensors need to have a length > 0 to calc mean squared error");

    tensor_new(n, 1, {1});
    tensor_view(&n_tensor, {1});
    var_from(n, n_tensor);
    var_from(sum, tensor_zeros(1));

    var_expr(diff, op(a, -, b));
    var_expr(squared, op(&diff, *, &diff));
    var_expr(acc_sum, op(&diff, <+>, &diff));
    var_expr(result, op(&acc_sum, /, &n));
    return result;
}
