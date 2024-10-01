#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

typedef enum { OP_LEAF, OP_ADD, OP_MUL } Op;

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
Tensor variable_forward(Variable *root);

#define backward(x) variable_backward((x))
void variable_backward(Variable *root);

Tensor chain_rule_mul(Variable *variable);
Tensor chain_rule_div(Variable *variable);

#define var_print(v, kind, ...)                                                \
    do {                                                                       \
        size_t v##_tensor_shape[] = __VA_ARGS__;                               \
        size_t v##_shape_len = ARR_LEN(v##_tensor_shape);                      \
        assert(v##_shape_len &&                                                \
               "Please specify the shape of the tensor view!");                \
        tensor_view(&v.kind, v##_tensor_shape);                                \
        printf("%s.%s = {\n", #v, #kind);                                      \
        {                                                                      \
            printf("\tshape = { ");                                            \
            {                                                                  \
                for (size_t i = 0; i < v##_shape_len; ++i) {                   \
                    if (i == v##_shape_len - 1)                                \
                        printf("%zu", v.kind.shape[i]);                        \
                    else                                                       \
                        printf("%zu, ", v.kind.shape[i]);                      \
                }                                                              \
            }                                                                  \
            printf(" }\n");                                                    \
            printf("\tdata = {\n");                                            \
            {                                                                  \
                int *v##_indices = (int *)malloc(v##_shape_len * sizeof(int)); \
                tensor_print(&v.kind, v##_shape_len, v##_indices, 0, "\t\t");  \
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
