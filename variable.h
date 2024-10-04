#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

typedef enum {
    OP_LEAF,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_DOT,
    OP_ACCUM_SUM,
    OP_SCALAR_SUM,
    OP_SCALAR_DIFF,
    OP_SCALAR_MUL,
    OP_SCALAR_POW,
} Op;

typedef struct Variable {
    struct Variable *left;
    struct Variable *right;
    Tensor items;
    Tensor grad;
    Op op;
} Variable;

#define op(left, op, right) variable_op((left), (#op), (right))
Variable variable_op(struct Variable *left, ...);

#define forward(x) variable_forward((x))
Variable variable_forward(Variable *root);

#define backward(x) variable_backward((x))
void variable_backward(Variable *root);

Tensor chain_rule_mul(Variable term, Variable constant);
Tensor chain_rule_div_numerator(Variable variable);
Tensor chain_rule_div_denominator(Variable left, Variable right);
Tensor chain_rule_pow(Variable base, Variable pow);
Tensor chain_rule_base_pow(Variable base, Variable pow);

#define var_print(v, kind, ...)                                                \
    do {                                                                       \
        printf("%s.%s = {\n", #v, #kind);                                      \
        {                                                                      \
            printf("\tdata = {\n");                                            \
            printf("\t\t");                                                    \
            {                                                                  \
                for (size_t i = 0; i < v.kind.length; ++i) {                   \
                    if (i == v.kind.length - 1) {                              \
                        printf("%f", v.kind.data[i]);                          \
                    } else {                                                   \
                        printf("%f, ", v.kind.data[i]);                        \
                    }                                                          \
                }                                                              \
            }                                                                  \
            printf("\n\t}\n");                                                 \
        }                                                                      \
        printf("}\n\n");                                                       \
    } while (0)

#define var_new(NAME, ...)                                                     \
    float NAME##_var_tensor_data[] = __VA_ARGS__;                              \
    tensor_new(NAME, ARR_LEN(NAME##_var_tensor_data), __VA_ARGS__);            \
    Variable NAME = variable_new(NAME##_tensor);
#define var_from(NAME, ...) Variable NAME = variable_new(__VA_ARGS__);
Variable variable_new(Tensor tensor);

#define var_expr(name, value) Variable name = (value)
#define var_rand(NAME, ...)                                                    \
    tensor_rand(NAME, __VA_ARGS__);                                            \
    var_from(NAME, NAME##_rand_tensor)

Variable var_copy(Variable variable, bool preserve_tree);
#define var_free(...)                                                          \
    do {                                                                       \
        Variable variables[] = {__VA_ARGS__};                                  \
        for (size_t i = 0; i < ARR_LEN(variables); ++i) {                      \
            Variable copy = var_copy(variables[i], false);                     \
            variables[i].op = OP_LEAF;                                         \
            free(variables[i].left);                                           \
            free(variables[i].right);                                          \
            tensor_free(variables[i].grad);                                    \
            tensor_free(variables[i].items);                                   \
            variables[i] = copy;                                               \
        }                                                                      \
    } while (0)
