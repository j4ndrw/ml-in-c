#include "tensor.h"

Tensor tensor_zeros(size_t length) {
    size_t default_shape[] = {length};
    Tensor zeros = {0};
    zeros.length = length;
    zeros.data = (float *)malloc(length * sizeof(float));
    zeros.shape = default_shape;
    for (size_t i = 0; i < length; ++i) {
        zeros.data[i] = 0;
    }
    return zeros;
}

Tensor tensor_ones(size_t length) {
    size_t default_shape[] = {length};
    Tensor ones = {0};
    ones.length = length;
    ones.data = (float *)malloc(length * sizeof(float));
    ones.shape = default_shape;
    for (size_t i = 0; i < length; ++i) {
        ones.data[i] = 1;
    }
    return ones;
}

Tensor *_tensor_view(Tensor *tensor, size_t shape[], size_t shape_size) {
    size_t reduced_dim = 0;
    for (size_t i = 0; i < shape_size; ++i) {
        if (reduced_dim == 0)
            reduced_dim = 1;
        reduced_dim *= shape[i];
    }
    assert(reduced_dim == tensor->length &&
           "Could not view tensor with given shape");
    tensor->shape = shape;
    return tensor;
}
