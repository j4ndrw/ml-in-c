#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "optimizer.h"
#include "variable.h"

void div_gradient_test() {
    var_new(a, {8});
    var_new(b, {2});
    var_expr(div, op(&a, /, &b));
    var_from(fwd, forward(&div));
    backward(&div);
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void multidim_dot_product_test() {
    var_rand(a, {1, 2, 3});
    var_rand(b, {1, 3, 4});
    var_expr(dot, op(&a, @, &b));
    var_from(result, forward(&dot));
    var_print(a, items, {});
    var_print(b, items, {});
    var_print(result, items, {1, 2, 4});
}

void dot_product_test() {
    var_new(a, {1, 2, 3, 4});
    var_new(id, {1, 1, 1, 1});

    tensor_view(&a.items, {2, 2});
    tensor_view(&id.items, {2, 2});

    var_expr(dot, op(&a, @, &id));
    var_from(result, forward(&dot));
    var_print(result, items, {2, 2});
}

void simple_backprop_test() {
    var_rand(a, {2, 2});
    var_rand(b, {2, 2});
    var_expr(c, op(&a, *, &b));

    float learning_rate = 0.1;
    SGDOptimizer optimizer = optimizer_sgd_create(&a, &b, learning_rate);

    var_from(result, forward(&c));

    printf("BEFORE BACKWARD:\n---------------\n");
    var_print(a, items, {2, 2});
    var_print(a, grad, {2, 2});
    var_print(b, items, {2, 2});
    var_print(b, grad, {2, 2});

    backward(&c);

    printf("AFTER BACKWARD:\n---------------\n");
    var_print(a, items, {2, 2});
    var_print(a, grad, {2, 2});
    var_print(b, items, {2, 2});
    var_print(b, grad, {2, 2});

    optimizer_sgd_step(&optimizer);
    printf("AFTER SGD STEP:\n---------------\n");
    var_print(a, items, {2, 2});
    var_print(a, grad, {2, 2});
    var_print(b, items, {2, 2});
    var_print(b, grad, {2, 2});

    optimizer_sgd_zero_grad(&optimizer);
    printf("AFTER SGD ZERO GRAD:\n---------------\n");
    var_print(a, items, {2, 2});
    var_print(a, grad, {2, 2});
    var_print(b, items, {2, 2});
    var_print(b, grad, {2, 2});
}

int main() {
    srand(time(NULL));
    div_gradient_test();
    return 0;
}
