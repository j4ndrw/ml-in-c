// c is hard bro...

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "loss.h"
#include "optimizer.h"
#include "tensor.h"
#include "variable.h"

void mse_test() {
    var_new(a, {44});
    var_new(b, {99});
    var_print(a, items, {});
    var_print(b, items, {});
    Variable loss = loss_mse(&a, &b);
    Variable graph = forward(&loss);
    backward(&graph);
    var_print(loss, items, {});
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void simple_neuron_test() {
    var_new(inputs, {1, 2, 3, 4, 5, 6});
    var_new(labels, {2, 4, 6, 8, 10, 12});

    var_rand(weights, {1});
    size_t epochs = 1000;
    float learning_rate = 0.01;
    SGDOptimizer optimizer =
        optimizer_sgd_create(&weights, NULL, learning_rate);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        optimizer_sgd_zero_grad(&optimizer);

        var_expr(prediction, op(&inputs, <*>, &weights));
        var_expr(loss, loss_mse(&labels, &prediction));

        Variable graph = forward(&loss);
        backward(&graph);
        optimizer_sgd_step(&optimizer);

        if (epoch % 100 == 0) {
            printf("EPOCH: %zu | LOSS: %f | WEIGHT: %f | WEIGHT_GRAD: %f\n",
                   epoch, loss.items.data[0], weights.items.data[0],
                   weights.grad.data[0]);
        }
    }

    var_free(inputs, labels, weights);
}

void mul_gradient_test() {
    var_new(a, {2});
    var_new(b, {3});
    var_expr(div, op(&a, *, &b));
    Variable graph = forward(&div);
    backward(&graph);
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void div_gradient_test() {
    var_new(a, {8});
    var_new(b, {2});
    var_expr(div, op(&a, /, &b));
    Variable graph = forward(&div);
    backward(&graph);
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void multidim_dot_product_test() {
    var_rand(a, {1, 2, 3});
    var_rand(b, {1, 3, 4});
    var_expr(dot, op(&a, @, &b));
    forward(&dot);
    var_print(a, items, {});
    var_print(b, items, {});
    var_print(dot, items, {1, 2, 4});
}

void dot_product_test() {
    tensor_new(a, 4, {1, 2, 3, 4});
    tensor_new(id, 4, {1, 1, 1, 1});

    tensor_view(a_tensor, {2, 2});
    tensor_view(id_tensor, {2, 2});

    var_from(a, a_tensor);
    var_from(id, id_tensor);

    var_expr(dot, op(&a, @, &id));
    forward(&dot);
    var_print(dot, items, {2, 2});
}

void simple_backprop_test() {
    var_rand(a, {2, 2});
    var_rand(b, {2, 2});
    var_expr(c, op(&a, *, &b));

    float learning_rate = 0.1;
    SGDOptimizer optimizer = optimizer_sgd_create(&a, &b, learning_rate);

    forward(&c);

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
    // simple_backprop_test();
    // dot_product_test();
    // div_gradient_test();
    mse_test();
    // simple_neuron_test();
    // mul_gradient_test();
    return 0;
}
