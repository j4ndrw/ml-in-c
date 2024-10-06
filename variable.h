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
    OP_SCALAR_DIV,
    OP_SCALAR_POW,
    OP_SCALAR_SQ,
} Op;

typedef struct Variable {
    struct Variable *left;
    struct Variable *right;
    Tensor items;
    Tensor grad;
    Op op;
    void (*backward)(struct Variable *root, struct Variable *left,
                     struct Variable *right);

} Variable;

#define op(left, op, right) variable_op((left), (#op), (right))
#define op_self(var, op) variable_op((var), (#op))
Variable variable_op(struct Variable *left, ...);

#define backward(x) variable_backward((x))
void variable_backward(Variable *root);

#define var_print(v, kind, ...)                                                \
    do {                                                                       \
        Tensor t = v.kind;                                                     \
        tensor_reset_shape(&t);                                                \
        size_t shape[] = __VA_ARGS__;                                          \
        size_t shape_len = ARR_LEN(shape);                                     \
        tensor_view(t, __VA_ARGS__);                                           \
        shape_len = tensor_shape_len(t);                                       \
        printf("%s.%s = {\n", #v, #kind);                                      \
        {                                                                      \
            printf("\tshape = { ");                                            \
            {                                                                  \
                size_t shape_idx = 0;                                          \
                while (t.shape[shape_idx] != 0) {                              \
                    printf("%zu, ", t.shape[shape_idx]);                       \
                    shape_idx++;                                               \
                }                                                              \
            }                                                                  \
            printf(" }\n");                                                    \
            printf("\tdata = {\n");                                            \
            {                                                                  \
                int *v##_indices = (int *)malloc(shape_len * sizeof(int));     \
                tensor_print(t, v##_indices, 0, "\t\t");                       \
            }                                                                  \
            printf("\n\t}\n");                                                 \
        }                                                                      \
        printf("}\n\n");                                                       \
    } while (0)

#define var_print_data(v, kind, ...)                                           \
    do {                                                                       \
        Tensor t = v.kind;                                                     \
        tensor_reset_shape(&t);                                                \
        size_t shape[] = __VA_ARGS__;                                          \
        size_t shape_len = ARR_LEN(shape);                                     \
        tensor_view(t, __VA_ARGS__);                                           \
        shape_len = tensor_shape_len(t);                                       \
        printf("%s.%s = { ", #v, #kind);                                       \
        for (size_t i = 0; i < t.length; ++i)                                  \
            printf("%f, ", t.data[i]);                                         \
        printf(" }\n");                                                        \
    } while (0)

#define var_new(NAME, ...)                                                     \
    double NAME##_var_tensor_data[] = __VA_ARGS__;                             \
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
