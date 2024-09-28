#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t length;
    size_t capacity;
    float *data;
} Tensor;

#define ARR_LEN(arr) ((int)sizeof(arr) / (int)sizeof((arr)[0]))

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

typedef enum { OP_LEAF, OP_ADD, OP_MUL, OP_SUB, OP_DIV } Op;
typedef struct Variable {
    struct Variable *left;
    struct Variable *right;
    Tensor items;
    Tensor grad;
    Op op;
} Variable;

Tensor tensor_new(float *data) {
    Tensor tensor = {0};
    for (size_t i = 0; i < ARR_LEN(data); ++i) {
        tensor_append(&tensor, data[i]);
    }
    return tensor;
}

Tensor tensor_empty() {
    Tensor tensor = {0};
    return tensor;
}

Tensor tensor_zeros(size_t length) {
    Tensor zeros = {0};
    zeros.length = length;
    for (size_t i = 0; i < length; ++i) {
        tensor_append(&zeros, 0);
    }
    return zeros;
}

Tensor tensor_ones(size_t length) {
    Tensor ones = {0};
    ones.length = length;
    for (size_t i = 0; i < length; ++i) {
        tensor_append(&ones, 1);
    }
    return ones;
}

#define VAR_INIT(name, value)                                                  \
    Variable name;                                                             \
    do {                                                                       \
        float data[] = value;                                                  \
        Variable name = variable_new(tensor_new(data));                        \
    } while (0)

#define VAR(name, value) Variable name = (value);

Variable variable_new(Tensor tensor) {
    Variable variable = {0};
    variable.left = NULL;
    variable.right = NULL;
    variable.op = OP_LEAF;
    variable.items = tensor;
    variable.grad = tensor_ones(tensor.length);
    return variable;
}

#define OP(left, op, right) variable_op((left), #op, (right))
Variable variable_op(struct Variable *left, ...) {
    va_list args;
    size_t argc = 2;
    va_start(args, argc);
    char *op_char = va_arg(args, char *);
    struct Variable *right = va_arg(args, struct Variable *);

    assert((op_char == "+" || op_char == "*" || op_char == "-" ||
            op_char == "/") &&
           "Invalid op_charerator");
    assert(right != NULL && "You cannot pass a null pointer when doing "
                            "operations on the computational graph");

    Op op;
    if (op_char == "+")
        op = OP_ADD;
    else if (op_char == "-")
        op = OP_SUB;
    else if (op_char == "*")
        op = OP_MUL;
    else if (op_char == "/")
        op = OP_DIV;
    else
        op = OP_LEAF;

    Tensor tensor = tensor_empty();
    Variable variable = {0};
    variable.left = left;
    variable.right = right;
    variable.op = op;
    variable.items = tensor;
    variable.grad = tensor_ones(tensor.length);

    va_end(args);
    return variable;
}

#define FORWARD(x) variable_forward((x))
Tensor variable_forward(Variable *root) {
    if (root->left == NULL) {
        return root->items;
    }

    if (root->op == OP_ADD) {
        assert(root->left->items.length == root->right->items.length &&
               "Cannot add items - Items have to have the same length");
        size_t length = root->left->items.length;
        Tensor result = {0};
        result.length = length;
        for (size_t i = 0; i < length; ++i) {
            tensor_append(&result, root->left->items.data[i] +
                                       root->right->items.data[i]);
        }
        return result;
    }

    if (root->op == OP_SUB) {
        assert(root->left->items.length == root->right->items.length &&
               "Cannot subtract items - Items have to have the same length");
        size_t length = root->left->items.length;
        Tensor result = {0};
        result.length = length;
        for (size_t i = 0; i < length; ++i) {
            tensor_append(&result, root->left->items.data[i] -
                                       root->right->items.data[i]);
        }
        return result;
    }

    if (root->op == OP_MUL) {
        assert(root->left->items.length == root->right->items.length &&
               "Cannot multiply items - Items have to have the same length");
        size_t length = root->left->items.length;
        Tensor result = {0};
        result.length = length;
        for (size_t i = 0; i < length; ++i) {
            tensor_append(&result, root->left->items.data[i] *
                                       root->right->items.data[i]);
        }
        return result;
    }

    if (root->op == OP_DIV) {
        assert(root->left->items.length == root->right->items.length &&
               "Cannot multiply items - Items have to have the same length");
        size_t length = root->left->items.length;
        Tensor result = {0};
        result.length = length;
        for (size_t i = 0; i < length; ++i) {
            assert(root->right->items.data[i] != 0.0f &&
                   "Division by zero - the hell you doin?");
            tensor_append(&result, root->left->items.data[i] /
                                       root->right->items.data[i]);
        }
        return result;
    }

    return tensor_empty();
}

int main() {
    VAR_INIT(a, {2.0f});
    VAR_INIT(b, {3.0f});
    VAR(c, OP(&a, +, &b));
    Tensor result = FORWARD(&c);
    printf("%zu\n", result.data[0]);
    return 0;
}
