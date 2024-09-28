#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

typedef enum { OP_LEAF, OP_ADD, OP_MUL, OP_SUB } Op;

typedef struct Variable {
    struct Variable *left;
    struct Variable *right;
    Tensor items;
    Tensor grad;
    Op op;
} Variable;

#define VAR_INIT(name, value)                                                  \
    float name##_data[] = value;                                                 \
    Variable name = variable_new(tensor_new(name##_data));

#define VAR(name, value) Variable name = (value);
Variable variable_new(Tensor tensor);

#define OP(left, op, right) variable_op((left), (#op), (right))
Variable variable_op(struct Variable *left, ...);

#define FORWARD(x) variable_forward((x))
Tensor variable_forward(Variable *root);

#define BACKWARD(x) variable_backward((x))
void variable_backward(Variable* root);

Tensor chain_rule_mul(Variable *variable);
Tensor chain_rule_div(Variable *variable);
