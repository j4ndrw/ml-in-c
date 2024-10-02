#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "loss.h"
#include "optimizer.h"
#include "variable.h"

void mse_test() {
    var_new(a, {5});
    var_new(b, {12});
    Variable loss = loss_mse(&a, &b);
    Tensor result = forward(&loss);
    tensor_print(result, {});
}

void simple_neuron_test() {
    var_new(inputs, {1, 2, 3, 4, 5, 6});
    var_new(labels, {2, 4, 6, 8, 10, 12});

    var_rand(weights, {1});
    size_t epochs = 100;
    float learning_rate = 0.0001;
    SGDOptimizer optimizer =
        optimizer_sgd_create(&weights, NULL, learning_rate);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0;
        for (size_t i = 0; i < inputs.items.length; ++i) {
            tensor_new(sample, 1, {inputs.items.data[i]});
            tensor_new(expected, 1, {labels.items.data[i]});

            var_from(expected, expected_tensor);

            var_from(sample, sample_tensor);
            var_expr(model, op(&sample, *, &weights));

            var_from(prediction, forward(&model));
            Variable loss = loss_mse(&prediction, &expected);
            Tensor loss_tensor = forward(&loss);
            total_loss += loss_tensor.data[0];

            backward(&loss);
        }
        optimizer_sgd_step(&optimizer);
        optimizer_sgd_zero_grad(&optimizer);
        printf("EPOCH: %zu | LOSS: %f\n", epoch, total_loss);
    }
}

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
    // simple_neuron_test();
    mse_test();
    return 0;
}
