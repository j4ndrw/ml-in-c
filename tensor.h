#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

typedef struct {
    float *data;
    size_t length;
    size_t capacity;
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

Tensor tensor_new(float *data);
Tensor tensor_empty();
Tensor tensor_zeros(size_t length);
Tensor tensor_ones(size_t length);
