#include "tensor.h"

Tensor tensor_new(float *data) {
    Tensor tensor = {0};
    size_t length = ARR_LEN(data);

    tensor.length = length;
    tensor.data = data;
    return tensor;
}

Tensor tensor_zeros(size_t length) {
    Tensor zeros = {0};
    zeros.length = length;
    zeros.data = (float *)malloc(length * sizeof(float));
    for (size_t i = 0; i < length; ++i) {
        zeros.data[i] = 0;
    }
    return zeros;
}

Tensor tensor_ones(size_t length) {
    Tensor ones = {0};
    ones.length = length;
    ones.data = (float *)malloc(length * sizeof(float));
    for (size_t i = 0; i < length; ++i) {
        ones.data[i] = 1;
    }
    return ones;
}
