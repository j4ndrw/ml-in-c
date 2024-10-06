#pragma once

#include "variable.h"

typedef struct {
  Variable* weights;
  Variable* biases;
  double learning_rate;
} SGDOptimizer;

SGDOptimizer optimizer_sgd_create(Variable* weights, Variable* biases, double learning_rate);
void optimizer_sgd_step(SGDOptimizer* optimizer);
void optimizer_sgd_zero_grad(SGDOptimizer* optimizer);
