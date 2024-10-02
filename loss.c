#include "tensor.h"
#include "variable.h"

Variable loss_mse(Variable a, Variable b) {
    assert(a.items.length > 0 &&
           "Tensors need to have a length > 0 to calc mean squared error");

    var_from(sum, tensor_new_scalar(0));
    var_from(n, tensor_new_scalar(a.items.length));

    var_expr(diff, op(&a, -, &b));
    var_expr(squared, op(&diff, *, &diff));
    var_expr(acc_sum, op(&sum, <+>, &squared));
    return op(&acc_sum, /, &n);
}
