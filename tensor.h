#include "utils.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *data;
    size_t length;
    size_t capacity;
    size_t *shape;
} Tensor;

// Credit: https://github.com/tsoding/panim/blob/main/src/nob.h
#define TENSOR_INIT_CAP 8192
#define tensor_append(tensor, item)                                            \
    do {                                                                       \
        if ((tensor)->length >= (tensor)->capacity) {                          \
            (tensor)->capacity = (tensor)->capacity == 0                       \
                                     ? TENSOR_INIT_CAP                         \
                                     : (tensor)->capacity * 2;                 \
            (tensor)->data = realloc(                                          \
                (tensor)->data, (tensor)->capacity * sizeof(*(tensor)->data)); \
            assert((tensor)->data != NULL && "Buy more RAM lol");              \
        }                                                                      \
        (tensor)->data[(tensor)->length++] = (item);                           \
    } while (0)

#define tensor_new(NAME, LENGTH, ...)                                          \
    Tensor NAME##_tensor = {0};                                                \
    float NAME##_data[] = __VA_ARGS__;                                         \
                                                                               \
    NAME##_tensor.data = NAME##_data;                                          \
    NAME##_tensor.length = LENGTH;                                             \
    NAME##_tensor.shape = (size_t *)malloc(sizeof(size_t));

Tensor tensor_zeros(size_t length);
Tensor tensor_ones(size_t length);

#define tensor_rand(NAME, ...)                                                 \
    size_t NAME##_rand_shape[] = __VA_ARGS__;                                  \
    Tensor NAME##_rand_tensor =                                                \
        _tensor_rand(NAME##_rand_shape, ARR_LEN(NAME##_rand_shape));
Tensor _tensor_rand(size_t shape[], size_t shape_len);

#define tensor_view(tensor, shape) _tensor_view(tensor, shape, ARR_LEN(shape))
void _tensor_view(Tensor *tensor, size_t shape[], size_t shape_len);

void tensor_print(Tensor *tensor, size_t shape_len, int *indices, int depth,
                  char *prefix);
