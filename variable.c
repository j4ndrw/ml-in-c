#include "variable.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Variable variable_new(Tensor tensor) {
    Variable variable = {0};
    variable.left = NULL;
    variable.right = NULL;
    variable.op = OP_LEAF;
    variable.items = tensor;
    variable.grad = tensor_ones(tensor.length);
    return variable;
}

Variable variable_op(struct Variable *left, ...) {
    va_list args;
    size_t argc = 2;
    va_start(args, argc);
    char *op_str = va_arg(args, char *);
    struct Variable *right = va_arg(args, struct Variable *);

    if (strcmp(op_str, "+") != 0 && strcmp(op_str, "-") != 0 &&
        strcmp(op_str, "*") != 0 && strcmp(op_str, "/") != 0) {
        fprintf(stderr, "Invalid op character. Found %s\n", op_str);
        exit(EXIT_FAILURE);
    }

    Op op;
    if (strcmp(op_str, "+") == 0)
        op = OP_ADD;
    else if (strcmp(op_str, "-") == 0)
        op = OP_SUB;
    else if (strcmp(op_str, "*") == 0)
        op = OP_MUL;
    else
        op = OP_LEAF;

    Tensor tensor = tensor_zeros(left->items.length);
    Variable variable = {0};
    variable.left = left;
    variable.right = right;
    variable.op = op;
    variable.items = tensor;
    variable.grad = tensor_ones(tensor.length);

    va_end(args);
    return variable;
}

Tensor chain_rule_mul(Variable *variable) {
    Tensor result = tensor_zeros(variable->grad.length);
    for (int i = 0; i < variable->grad.length; ++i) {
        result.data[i] = variable->items.data[i] * variable->grad.data[i];
    }
    return result;
}

Tensor variable_forward(Variable *root) {
    if (root->left == NULL) {
        return root->items;
    }

    assert(root->left->items.length == root->right->items.length &&
           "Cannot perform op - Items have to have the same length");

    size_t length = root->left->items.length;
    Tensor result = tensor_zeros(length);

    if (root->op == OP_ADD) {
        for (size_t i = 0; i < length; ++i) {
            result.data[i] =
                root->left->items.data[i] + root->right->items.data[i];
        }
        return result;
    }

    if (root->op == OP_SUB) {
        for (size_t i = 0; i < length; ++i) {
            result.data[i] =
                root->left->items.data[i] - root->right->items.data[i];
        }
        return result;
    }

    if (root->op == OP_MUL) {
        for (size_t i = 0; i < length; ++i) {
            result.data[i] =
                root->left->items.data[i] * root->right->items.data[i];
        }
        return result;
    }

    return result;
}

void variable_backward(Variable *root) {
    if (root->left == NULL)
        return;

    switch (root->op) {
    case OP_ADD:
    case OP_SUB: {
        root->left->grad = tensor_ones(root->left->grad.length);
        root->right->grad = tensor_ones(root->right->grad.length);
    } break;
    case OP_MUL: {
        Tensor left_grad = chain_rule_mul(root->right);
        Tensor right_grad = chain_rule_mul(root->left);
        root->left->grad = left_grad;
        root->right->grad = right_grad;
    } break;
    }

    variable_backward(root->left);
    variable_backward(root->right);
}
