#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "optimizer.h"
#include "variable.h"

int main() {
    srand(time(NULL));

    var_rand(a, {4, 4});
    var_rand(b, {4, 4});
    var_expr(c, op(&a, *, &b));

    SGDOptimizer optimizer = optimizer_sgd_create(&a, &b, 0.0001);

    var_from(result, forward(&c));

    printf("BEFORE BACKWARD:\n---------------\n");
    var_print(a, items, {4, 4});
    var_print(a, grad, {4, 4});
    var_print(b, items, {4, 4});
    var_print(b, grad, {4, 4});

    backward(&c);

    printf("AFTER BACKWARD:\n---------------\n");
    var_print(a, items, {4, 4});
    var_print(a, grad, {4, 4});
    var_print(b, items, {4, 4});
    var_print(b, grad, {4, 4});

    optimizer_sgd_step(&optimizer);
    printf("AFTER SGD STEP:\n---------------\n");
    var_print(a, items, {4, 4});
    var_print(a, grad, {4, 4});
    var_print(b, items, {4, 4});
    var_print(b, grad, {4, 4});

    optimizer_sgd_zero_grad(&optimizer);
    printf("AFTER SGD ZERO GRAD:\n---------------\n");
    var_print(a, items, {4, 4});
    var_print(a, grad, {4, 4});
    var_print(b, items, {4, 4});
    var_print(b, grad, {4, 4});
    return 0;
}
