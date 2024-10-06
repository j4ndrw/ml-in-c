// c is hard bro...

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "activation.h"
#include "calc.h"
#include "loss.h"
#include "optimizer.h"
#include "tensor.h"
#include "variable.h"

void mse_test() {
    var_new(inputs, {44});
    var_new(labels, {99});
    loss_mse(loss, inputs, labels);
    backward(&loss);
    var_print(inputs, grad, {});
    var_print(labels, grad, {});
}

void simple_neuron_linear_regression_test() {
    // Data
    var_new(inputs, {1, 2, 3, 4, 5, 6, 7, 8});
    var_new(labels, {8, 16, 24, 32, 40, 48, 56, 64});

    // Hyperparams
    size_t epochs = 10000;
    double learning_rate = 1e-3;

    // Model
    var_rand(weights, {1});

    // Training
    SGDOptimizer optimizer =
        optimizer_sgd_create(&weights, NULL, learning_rate);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        optimizer_sgd_zero_grad(&optimizer);

        var_expr(prediction, op(&inputs, <*>, &weights));
        loss_mse(loss, labels, prediction);

        backward(&loss);

        // var_print(inputs, grad, {});
        optimizer_sgd_step(&optimizer);

        if (epoch % 100 == 0) {
            printf("EPOCH: %zu | LOSS: %f | WEIGHT: %f | WEIGHT_GRAD: %f\n",
                   epoch, loss.items.data[0], weights.items.data[0],
                   weights.grad.data[0]);
        }
    }

    // Validation
    var_new(test, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    var_expr(predictions, op(&test, <*>, &weights));
    var_print(test, items, {});
    var_print(predictions, items, {});
}

void xor_test() {
    // Data
    var_new(inputs, {0, 0, 0, 1, 1, 0, 1, 1});
    var_new(labels, {0, 1, 1, 0});
    tensor_view(inputs.items, {4, 2});

    // var_print(inputs, items, {4, 2});
    // var_print(labels, items, {});

    // Hyperparams
    size_t epochs = 1000;
    double learning_rate = 1e-3;

    // Model
    var_rand(w1, {1});
    var_rand(w2, {1});

    // Training
    SGDOptimizer optimizerw1 = optimizer_sgd_create(&w1, NULL, learning_rate);
    SGDOptimizer optimizerw2 = optimizer_sgd_create(&w2, NULL, learning_rate);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0f;
        for (size_t batch = 0; batch < inputs.items.shape[0]; ++batch) {
            optimizer_sgd_zero_grad(&optimizerw1);
            optimizer_sgd_zero_grad(&optimizerw2);
            tensor_new(batch_inputs, inputs.items.shape[1],
                       {inputs.items.data[0 * inputs.items.shape[1] + batch],
                        inputs.items.data[1 * inputs.items.shape[1] + batch]});

            var_from(batch_inputs, batch_inputs_tensor);
            var_expr(y1, op(&batch_inputs, *, &w1));
            var_expr(y2, op(&y1, *, &w1));
            activation_sigmoid(a1, y1);
            activation_sigmoid(a2, y2);
            calc_avg_2(prediction, a1, a2);
            loss_mse(loss, labels, prediction);
            backward(&loss);
            optimizer_sgd_step(&optimizerw1);
            optimizer_sgd_step(&optimizerw2);
        }
        if (epoch % 100 == 0) {
            printf("EPOCH: %zu | LOSS: %f | ", epoch, total_loss);
        }
    }

    // Validation
    var_new(t1, {0, 0});
    var_expr(y1, op(&t1, <*>, &w1));
    var_expr(y2, op(&y1, <*>, &w1));
    activation_sigmoid(a1, y1);
    activation_sigmoid(a2, y2);
    calc_avg_2(p1, y1, y2);
    var_print(t1, items, {});
    var_print(p1, items, {});
}

void mul_gradient_test() {
    var_new(a, {2});
    var_new(b, {3});
    var_expr(div, op(&a, *, &b));
    backward(&div);
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void div_gradient_test() {
    var_new(a, {8});
    var_new(b, {2});
    var_expr(div, op(&a, /, &b));
    backward(&div);
    var_print(div, items, {});
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void pow_gradient_test() {
    var_new(a, {8});
    var_new(b, {2});
    var_expr(pow, op(&a, <^>, &b));
    backward(&pow);
    var_print(pow, items, {});
    var_print(a, grad, {});
    var_print(b, grad, {});
}

void multidim_dot_product_test() {
    var_rand(a, {1, 2, 3});
    var_rand(b, {1, 3, 4});
    var_expr(dot, op(&a, @, &b));
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
    var_print(dot, items, {2, 2});
}

void simple_backprop_test() {
    var_rand(a, {2, 2});
    var_rand(b, {2, 2});
    var_expr(c, op(&a, *, &b));

    double learning_rate = 0.1;
    SGDOptimizer optimizer = optimizer_sgd_create(&a, &b, learning_rate);

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
    // mse_test();
    simple_neuron_linear_regression_test();
    // xor_test();
    // mul_gradient_test();
    // div_gradient_test();
    // pow_gradient_test();
    // multidim_dot_product_test();
    // dot_product_test();
    // simple_backprop_test();
    return 0;
}
