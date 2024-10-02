#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "tensor.h"
#include "utils.h"
#include "variable.h"

Tensor tensor_zeros(size_t length) {
    tensor_new(zeros, length, {0});
    zeros_tensor.data = (float *)malloc(length * sizeof(float));
    zeros_tensor.shape.data[0] = length;

    for (size_t i = 0; i < length; ++i) {
        zeros_tensor.data[i] = 0;
    }
    return zeros_tensor;
}

Tensor tensor_ones(size_t length) {
    tensor_new(ones, length, {0});
    ones_tensor.data = (float *)malloc(length * sizeof(float));
    ones_tensor.shape.data[0] = length;

    for (size_t i = 0; i < length; ++i) {
        ones_tensor.data[i] = 1;
    }
    return ones_tensor;
}

Tensor _tensor_view(Tensor tensor, size_t shape[], size_t shape_len) {
    if (shape_len == 0)
        return tensor;
    size_t reduced_dim = 0;
    for (size_t i = 0; i < shape_len; ++i) {
        if (reduced_dim == 0)
            reduced_dim = 1;
        reduced_dim *= shape[i];
    }
    if (reduced_dim == 0)
        return tensor;

    assert(reduced_dim == tensor.length &&
           "Could not view tensor with given shape");

    Tensor new_tensor = tensor_zeros(tensor.length);
    new_tensor.shape.data = realloc(new_tensor.shape.data, shape_len * sizeof(size_t));
    new_tensor.shape.length = shape_len;
    for (size_t i = 0; i < shape_len; ++i) {
        new_tensor.shape.data[i] = shape[i];
    }
    for (size_t i = 0; i < tensor.length; ++i) {
        new_tensor.data[i] = tensor.data[i];
    }
    return new_tensor;
}

// Stub generated by an LLM, but then perfected.
// I couldn't be asked to reason about this - too primitive of a functionality
// to matter
void __tensor_print(Tensor tensor, int *indices, int depth, char *prefix) {
    if (depth == tensor.shape.length - 1) {
        printf("%s[", prefix);
        for (int i = 0; i < tensor.shape.data[depth]; i++) {
            indices[depth] = i;
            int index = 0;
            for (int dim_idx = 0; dim_idx < tensor.shape.length; dim_idx++) {
                index = index * tensor.shape.data[dim_idx] + indices[dim_idx];
            }
            printf("%f", tensor.data[index]);
            if (i < tensor.shape.data[depth] - 1) {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    printf("%s[\n", prefix);
    for (int i = 0; i < tensor.shape.data[depth]; i++) {
        indices[depth] = i;
        for (int j = 0; j < depth + 1; j++) {
            printf("\t");
        }
        __tensor_print(tensor, indices, depth + 1, prefix);
        if (i < tensor.shape.data[depth] - 1) {
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
        rand_tensor.data[i] = randf();
    }
    _tensor_view(rand_tensor, shape, shape_len);
    return rand_tensor;
}

Tensor tensor_add(Tensor a, Tensor b) {
    assert(a.length == b.length && "Tensors need to have the same length!");
    size_t length = a.length;
    Tensor result = tensor_zeros(a.length);
    for (size_t i = 0; i < length; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

Tensor tensor_sub(Tensor a, Tensor b) {
    assert(a.length == b.length && "Tensors need to have the same length!");
    size_t length = a.length;
    Tensor result = tensor_zeros(a.length);
    for (size_t i = 0; i < length; ++i) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

Tensor tensor_mul(Tensor a, Tensor b) {
    assert(a.length == b.length && "Tensors need to have the same length!");
    size_t length = a.length;
    Tensor result = tensor_zeros(a.length);
    for (size_t i = 0; i < length; ++i) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}

Tensor tensor_div(Tensor a, Tensor b) {
    assert(a.length == b.length && "Tensors need to have the same length!");
    size_t length = a.length;
    Tensor result = tensor_zeros(a.length);
    for (size_t i = 0; i < length; ++i) {
        assert(b.data[i] > 0 && "Division by zero!");
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
}

Tensor tensor_dot(Tensor a, Tensor b) {
    if (a.shape.length == 1 && b.shape.length == 1)
        return tensor_mul(a, b);

    assert(a.shape.data[a.shape.length - 1] ==
               b.shape.data[b.shape.length - 2] &&
           "The inner sizes must be equal for the dot product to be possible");

    size_t n = a.shape.data[a.shape.length - 1];

    size_t rows = 1;
    for (size_t i = 0; i < a.shape.length - 1; ++i) {
        rows *= a.shape.data[i];
    }
    size_t cols = b.shape.data[a.shape.length - 1];

    Tensor result = tensor_zeros(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < n; ++k) {
                result.data[i * cols + j] +=
                    a.data[i * n + k] * b.data[k * cols + j];
            }
        }
    }
    return result;
}

Tensor tensor_scalar_accumulate(Tensor accumulator, Tensor t) {
    assert(accumulator.length == 1 && "The accumulator must be a scalar value");
    size_t length = t.length;
    Tensor result = tensor_zeros(1);
    for (size_t i = 0; i < length; ++i) {
        result.data[0] += accumulator.data[0] + t.data[i];
    }
    return result;
}

Tensor tensor_new_scalar(float value) {
    tensor_new(scalar, 1, {1});
    return scalar_tensor;
}

void tensor_reset_shape(Tensor *t) {
    t->shape = (Shape){.length = 1, .data = (size_t *)malloc(sizeof(size_t))};
    size_t shape[] = {t->length};
    t->shape.data = shape;
}
