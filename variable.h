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

#define var_init(NAME, LENGTH, ...)                                            \
    tensor_new(NAME, LENGTH, __VA_ARGS__);                                     \
    Variable NAME = variable_new(NAME##_tensor);

#define var_expr(name, value) Variable name = (value)
Variable variable_new(Tensor tensor);

#define op(left, op, right) variable_op((left), (#op), (right))
Variable variable_op(struct Variable *left, ...);

#define forward(x) variable_forward((x))
Tensor variable_forward(Variable *root);

#define backward(x) variable_backward((x))
void variable_backward(Variable *root);

Tensor chain_rule_mul(Variable *variable);
Tensor chain_rule_div(Variable *variable);

#define variable_print(v, kind, ...)                                           \
    size_t v##_tensor_shape[] = {__VA_ARGS__};                                 \
    if (ARR_LEN(v##_tensor_shape) > 0)                                         \
        v.kind = *tensor_view(&v.kind, v##_tensor_shape);                      \
    printf("%s.%s = {\n", #v, #kind);                                          \
    {                                                                          \
        printf("\tshape = { ");                                                \
        {                                                                      \
                                                                               \
            for (size_t i = 0; i < ARR_LEN(v##_tensor_shape); ++i) {           \
                if (i == ARR_LEN(v##_tensor_shape) - 1)                        \
                    printf("%zu", v.kind.shape[i]);                            \
                else                                                           \
                    printf("%zu, ", v.kind.shape[i]);                          \
            }                                                                  \
        }                                                                      \
        printf(" }\n");                                                        \
        printf("\tdata = {\n");                                                \
        {                                                                      \
            for (size_t i = 0; i < ARR_LEN(v##_tensor_shape); ++i) {           \
                printf("\t\t[\n");                                             \
                for (size_t j = 0; j < v##_tensor_shape[i]; ++j) {             \
                    printf("\t\t\t%f ", v.kind.data[j]);                       \
                }                                                              \
                printf("\n\t\t]\n");                                           \
            }                                                                  \
        }                                                                      \
        printf(" }\n");                                                        \
    }                                                                          \
    printf("}\n\n");
