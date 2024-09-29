#include "tensor.h"
#include "utils.h"

Tensor tensor_zeros(size_t length) {
    tensor_new(zeros, length, {0});
    zeros_tensor.data = (float *)malloc(length * sizeof(float));
    zeros_tensor.shape[0] = length;

    for (size_t i = 0; i < length; ++i) {
        zeros_tensor.data[i] = 0;
    }
    return zeros_tensor;
}

Tensor tensor_ones(size_t length) {
    tensor_new(ones, length, {0});
    ones_tensor.data = (float *)malloc(length * sizeof(float));
    ones_tensor.shape[0] = length;

    for (size_t i = 0; i < length; ++i) {
        ones_tensor.data[i] = 1;
    }
    return ones_tensor;
}

void _tensor_view(Tensor *tensor, size_t shape[], size_t shape_len) {
    size_t reduced_dim = 0;
    for (size_t i = 0; i < shape_len; ++i) {
        if (reduced_dim == 0)
            reduced_dim = 1;
        reduced_dim *= shape[i];
    }
    if (reduced_dim == 0)
        return;
    assert(reduced_dim == tensor->length &&
           "Could not view tensor with given shape");
    tensor->shape = realloc(tensor->shape, shape_len * sizeof(shape[0]));
    tensor->shape = shape;
}

// Stub generated by an LLM, but then perfected.
// I couldn't be asked to reason about this - too primitive of a functionality
// to matter
void tensor_print(Tensor *tensor, size_t shape_len, int *indices, int depth,
                  char *prefix) {
    if (depth == shape_len - 1) {
        printf("%s[", prefix);
        for (int i = 0; i < tensor->shape[depth]; i++) {
            indices[depth] = i;
            int index = 0;
            for (int dim_idx = 0; dim_idx < shape_len; dim_idx++) {
                index = index * tensor->shape[dim_idx] + indices[dim_idx];
            }
            printf("%f", tensor->data[index]);
            if (i < tensor->shape[depth] - 1) {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    printf("%s[\n", prefix);
    for (int i = 0; i < tensor->shape[depth]; i++) {
        indices[depth] = i;
        for (int j = 0; j < depth + 1; j++) {
            printf("\t");
        }
        tensor_print(tensor, shape_len, indices, depth + 1, prefix);
        if (i < tensor->shape[depth] - 1) {
            printf(",\n");
        }
    }
    printf("\n");
    for (int j = 0; j < depth; j++) {
        printf("\t");
    }
    printf("%s]", prefix);
}

Tensor _tensor_rand(size_t shape[], size_t shape_len) {
    size_t length = 0;
    for (size_t i = 0; i < shape_len; i++) {
        if (length == 0)
            length = 1;
        length *= shape[i];
    }
    assert(length && "Cannot initialize a zero or negative-shaped tensor!");
    Tensor rand_tensor = tensor_zeros(length);
    for (size_t i = 0; i < length; ++i) {
        rand_tensor.data[i] = randf() / 2.0f;
    }
    _tensor_view(&rand_tensor, shape, shape_len);
    return rand_tensor;
}
