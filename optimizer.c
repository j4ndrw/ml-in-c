#include "optimizer.h"

SGDOptimizer optimizer_sgd_create(Variable *weights, Variable *biases,
                           float learning_rate) {
    SGDOptimizer optimizer = {0};
    optimizer.weights = weights;
    optimizer.biases = biases;
    optimizer.learning_rate = learning_rate;
    return optimizer;
}

void optimizer_sgd_step(SGDOptimizer *optimizer) {
    if (optimizer->weights != NULL) {
        for (size_t i = 0; i < optimizer->weights->items.length; ++i) {
            optimizer->weights->items.data[i] -=
                optimizer->learning_rate * optimizer->weights->grad.data[i];
        }
    }

    if (optimizer->biases != NULL) {
        for (size_t i = 0; i < optimizer->biases->items.length; ++i) {
            optimizer->biases->items.data[i] -=
                optimizer->learning_rate * optimizer->biases->grad.data[i];
        }
    }
}

void optimizer_sgd_zero_grad(SGDOptimizer *optimizer) {
    if (optimizer->weights != NULL) {
        optimizer->weights->grad = tensor_ones(optimizer->weights->items.length);
    }

    if (optimizer->biases != NULL) {
        optimizer->biases->grad = tensor_ones(optimizer->biases->items.length);
    }
}
