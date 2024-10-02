#pragma once

#include <assert.h>
#include <stdarg.h>
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

#define backward(x) variable_backward((x))
void variable_backward(Variable *root);

Tensor chain_rule_mul(Variable variable);
Tensor chain_rule_div_numerator(Variable variable);
Tensor chain_rule_div_denominator(Variable left, Variable right);

#define var_print(v, kind, ...)                                                \
    do {                                                                       \
        Tensor v##_kind = v.kind;                                              \
        tensor_view(v##_kind, __VA_ARGS__);                                    \
        Tensor v##_view = v##_kind_view;                                       \
        printf("%s.%s = {\n", #v, #kind);                                      \
        {                                                                      \
            printf("\tshape = { ");                                            \
            {                                                                  \
                for (size_t i = 0; i < v##_view.shape.length; ++i) {      \
                    if (i == v##_view.shape.length - 1)                   \
                        printf("%zu", v##_view.shape.data[i]);          \
                    else                                                       \
                        printf("%zu, ", v##_view.shape.data[i]);        \
                }                                                              \
            }                                                                  \
            printf(" }\n");                                                    \
            printf("\tdata = {\n");                                            \
            {                                                                  \
                int *v##_indices =                                             \
                    (int *)malloc(v##_view.shape.length * sizeof(int));   \
                __tensor_print(v##_view, v##_indices, 0, "\t\t");         \
            }                                                                  \
            printf("\n\t}\n");                                                 \
        }                                                                      \
        printf("}\n\n");                                                       \
    } while (0)

#define var_print_ptr(v, kind, ...)                                            \
    do {                                                                       \
        tensor_view(&v->kind, __VA_ARGS__);                                    \
        printf("%s.%s = {\n", #v, #kind);                                      \
        {                                                                      \
            printf("\tshape = { ");                                            \
            {                                                                  \
                for (size_t i = 0; i < v->kind.shape.length; ++i) {            \
                    if (i == v->kind.shape.length - 1)                         \
                        printf("%zu", v->kind.shape.data[i]);                  \
                    else                                                       \
                        printf("%zu, ", v->kind.shape.data[i]);                \
                }                                                              \
            }                                                                  \
            printf(" }\n");                                                    \
            printf("\tdata = {\n");                                            \
            {                                                                  \
                int *v##_indices =                                             \
                    (int *)malloc(v->kind.shape.length * sizeof(int));         \
                tensor_print(&v->kind, v##_indices, 0, "\t\t");                \
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
