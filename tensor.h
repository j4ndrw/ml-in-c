#pragma once

#include "utils.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t *data;
    size_t length;
} Shape;

typedef struct {
    float *data;
    size_t length;
    Shape shape;
} Tensor;

#define tensor_new(NAME, LENGTH, ...)                                          \
    Tensor NAME##_tensor = {0};                                                \
    float NAME##_data[] = __VA_ARGS__;                                         \
                                                                               \
    NAME##_tensor.data = NAME##_data;                                          \
    NAME##_tensor.length = LENGTH;                                             \
    NAME##_tensor.shape.data = (size_t *)malloc(sizeof(size_t));               \
    NAME##_tensor.shape.length = 1;

Tensor tensor_zeros(size_t length);
Tensor tensor_ones(size_t length);

#define tensor_rand(NAME, ...)                                                 \
    size_t NAME##_rand_shape[] = __VA_ARGS__;                                  \
    Tensor NAME##_rand_tensor =                                                \
        _tensor_rand(NAME##_rand_shape, ARR_LEN(NAME##_rand_shape));
Tensor _tensor_rand(size_t shape[], size_t shape_len);

#define tensor_view(tensor, ...)                                               \
    Tensor tensor##_view;                                                      \
    do {                                                                       \
        size_t shape[] = __VA_ARGS__;                                          \
        tensor##_view = __tensor_view(tensor, shape);                          \
    } while (0)
#define __tensor_view(tensor, shape) _tensor_view(tensor, shape, ARR_LEN(shape))
Tensor _tensor_view(Tensor tensor, size_t shape[], size_t shape_len);

#define tensor_print_ptr(t, ...)                                               \
    do {                                                                       \
        Tensor t##_ptr = *t;                                                   \
        tensor_print(t##_ptr, __VA_ARGS__);                                    \
    } while (0)
#define tensor_print(t, ...)                                                   \
    do {                                                                       \
        var_from(t##_tensor, t);                                               \
        var_print(t##_tensor, items, __VA_ARGS__);                             \
    } while (0)
void __tensor_print(Tensor tensor, int *indices, int depth, char *prefix);

Tensor tensor_add(Tensor a, Tensor b);
Tensor tensor_sub(Tensor a, Tensor b);
Tensor tensor_mul(Tensor a, Tensor b);
Tensor tensor_div(Tensor a, Tensor b);
Tensor tensor_dot(Tensor a, Tensor b);
Tensor tensor_scalar_accumulate(Tensor accumulator, Tensor t);

Tensor tensor_new_scalar(float value);

void tensor_reset_shape(Tensor *t);
