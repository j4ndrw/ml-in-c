#include "variable.h"

#define loss_mse(result, actual, predicted)                                    \
    var_from(loss_mse_pow_n, SNEW(actual.items.length));                       \
    var_expr(diff, op(&actual, -, &predicted));                                \
    var_expr(squared, op_self(&diff, **));                                     \
    var_expr(loss_mse_sum, op_self(&squared, [+]));                            \
    var_expr(result, op(&loss_mse_sum, /, &loss_mse_pow_n));
