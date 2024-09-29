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
    size_t v##_tensor_shape[] = __VA_ARGS__;                                 \
    size_t shape_len = ARR_LEN(v##_tensor_shape);                              \
    assert(shape_len && "Please specify the shape of the tensor view!");   \
    tensor_view(&v.kind, v##_tensor_shape);                                    \
    printf("%s.%s = {\n", #v, #kind);                                          \
    {                                                                          \
        printf("\tshape = { ");                                                \
        {                                                                      \
            for (size_t i = 0; i < shape_len; ++i) {                           \
                if (i == shape_len - 1)                                        \
                    printf("%zu", v.kind.shape[i]);                            \
                else                                                           \
                    printf("%zu, ", v.kind.shape[i]);                          \
            }                                                                  \
        }                                                                      \
        printf(" }\n");                                                        \
        printf("\tdata = {\n");                                                \
        {                                                                      \
            int *indices = (int *)malloc(shape_len * sizeof(int));             \
            tensor_print(&v.kind, shape_len, indices, 0, "\t\t");              \
        }                                                                      \
        printf("\n\t}\n");                                                     \
    }                                                                          \
    printf("}\n\n");

#define var_new(NAME, LENGTH, ...)                                             \
    tensor_new(NAME, LENGTH, __VA_ARGS__);                                     \
    Variable NAME = variable_new(NAME##_tensor);
#define var_from(NAME, LENGTH, __VA_ARGS__)                                    \
    Variable NAME = variable_new(__VA_ARGS__);
Variable variable_new(Tensor tensor);

#define var_expr(name, value) Variable name = (value)
#define var_rand(NAME, LENGTH, ...)                                            \
    tensor_rand(NAME, __VA_ARGS__);                                            \
    var_from(NAME, LENGTH, NAME##_rand_tensor)
