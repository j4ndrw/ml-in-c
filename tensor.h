#pragma once

#include "utils.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// idfk why you would want a 1024-dimensional tensor, but there you go...
#define MAX_DIMS 1024

typedef struct {
    double *data;
    size_t length;
    size_t shape[MAX_DIMS];
} Tensor;

// Allocations
#define tensor_new(NAME, LENGTH, ...)                                          \
    Tensor NAME##_tensor;                                                      \
    do {                                                                       \
        double NAME##_data[] = __VA_ARGS__;                                    \
        NAME##_tensor.data = (double *)malloc(LENGTH * sizeof(double));        \
        assert(NAME##_tensor.data != NULL && "Memory allocation failed");      \
        for (size_t i = 0; i < LENGTH; ++i) {                                  \
            NAME##_tensor.data[i] = NAME##_data[i];                            \
        }                                                                      \
        NAME##_tensor.length = LENGTH;                                         \
        NAME##_tensor.shape[0] = LENGTH;                                       \
    } while (0)
Tensor tensor_empty(size_t length);
Tensor tensor_zeros(size_t length);
Tensor tensor_ones(size_t length);
Tensor tensor_from(size_t length, double value);
Tensor tensor_new_scalar(double value);
void tensor_reset_shape(Tensor *t);
Tensor tensor_copy(Tensor tensor);
void tensor_free(Tensor tensor);

// Meta
void tensor_print(Tensor tensor, int *indices, int depth, char *prefix);
#define tensor_rand(NAME, ...)                                                 \
    size_t NAME##_rand_shape[] = __VA_ARGS__;                                  \
    Tensor NAME##_rand_tensor =                                                \
        _tensor_rand(NAME##_rand_shape, ARR_LEN(NAME##_rand_shape));
Tensor _tensor_rand(size_t shape[], size_t shape_len);

#define tensor_view_from_shape(tensor, shape, shape_len)                       \
    do {                                                                       \
        if (shape_len == 0)                                                    \
            break;                                                             \
                                                                               \
        size_t reduced_dim = 0;                                                \
        for (size_t i = 0; i < shape_len; ++i) {                               \
            if (reduced_dim == 0)                                              \
                reduced_dim = 1;                                               \
            reduced_dim *= shape[i];                                           \
        }                                                                      \
        if (reduced_dim == 0)                                                  \
            break;                                                             \
                                                                               \
        assert(reduced_dim == tensor.length &&                                 \
               "Could not view tensor with given shape");                      \
                                                                               \
        tensor_reset_shape(&tensor);                                           \
        for (size_t i = 0; i < shape_len; ++i) {                               \
            tensor.shape[i] = shape[i];                                        \
        }                                                                      \
    } while (0)
#define tensor_view(tensor, ...)                                               \
    do {                                                                       \
        size_t shape[] = __VA_ARGS__;                                          \
        size_t shape_len = ARR_LEN(shape);                                     \
        tensor_view_from_shape(tensor, shape, shape_len);                      \
    } while (0)
size_t tensor_shape_len(Tensor tensor);

// Ops
Tensor tensor_add(Tensor a, Tensor b);
Tensor tensor_sub(Tensor a, Tensor b);
Tensor tensor_mul(Tensor a, Tensor b);
Tensor tensor_div(Tensor a, Tensor b);
Tensor tensor_dot(Tensor a, Tensor b);
Tensor tensor_sum(Tensor t);
Tensor tensor_pow(Tensor tensor, Tensor pow);
Tensor tensor_scalar_sum(Tensor t, Tensor scalar);
Tensor tensor_scalar_diff(Tensor t, Tensor scalar);
Tensor tensor_scalar_mul(Tensor t, Tensor scalar);
Tensor tensor_scalar_pow(Tensor base, Tensor pow);
Tensor tensor_scalar_sq(Tensor tensor);
Tensor tensor_natural_log(Tensor t);

// Derivatives
#define chain_rule_add(result, grad) Tensor result = (grad)

#define chain_rule_mul(result, grad, tensor)                                   \
    Tensor result = tensor_zeros((grad).length);                               \
    do {                                                                       \
        for (size_t i = 0; i < (grad).length; ++i) {                           \
            result.data[i] = (tensor).data[i] * (grad).data[i];                \
        }                                                                      \
    } while (0)

#define chain_rule_div_numerator(result, grad, tensor)                         \
    Tensor result = tensor_div((grad), (tensor));

#define chain_rule_div_denominator(result, grad, left, right)                  \
    Tensor result = tensor_mul(                                                \
        (grad), tensor_div(tensor_scalar_mul(left, tensor_new_scalar(-1)),     \
                           tensor_scalar_sq(right)))

#define chain_rule_pow(result, grad, left, right)                              \
    Tensor result = tensor_mul(                                                \
        (grad),                                                                \
        tensor_mul((right),                                                    \
                   tensor_pow((left), tensor_scalar_diff(                      \
                                          (right), tensor_new_scalar(1)))));

#define chain_rule_base_pow(result, grad, left, right)                         \
    Tensor result =                                                            \
        tensor_mul((grad), tensor_mul(tensor_pow((left), (right)),             \
                                      tensor_natural_log((left))));
