#include "variable.h"
#include "graph.h"
#include "tensor.h"

#include <stdarg.h>
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

Variable variable_op(Variable *left, ...) {
    va_list args;
    va_start(args, left);

    char *op_str = va_arg(args, char *);
    Variable *right = va_arg(args, Variable *);

    if (strcmp(op_str, "+") != 0 && strcmp(op_str, "-") != 0 &&
        strcmp(op_str, "*") != 0 && strcmp(op_str, "/") != 0 &&
        strcmp(op_str, "@") != 0 && strcmp(op_str, "<+>") != 0) {
        fprintf(stderr,
                "Invalid op character. Expected one of {'+', '-', '*', '/', "
                "'@'}, but found %s\n",
                op_str);
        exit(EXIT_FAILURE);
    }

    Op op;
    if (strcmp(op_str, "+") == 0)
        op = OP_ADD;
    else if (strcmp(op_str, "-") == 0)
        op = OP_SUB;
    else if (strcmp(op_str, "*") == 0)
        op = OP_MUL;
    else if (strcmp(op_str, "/") == 0)
        op = OP_DIV;
    else if (strcmp(op_str, "@") == 0)
        op = OP_DOT;
    else if (strcmp(op_str, "<+>") == 0)
        op = OP_ACCUM_SUM;
    else
        op = OP_LEAF;

    Variable variable;
    variable.left = left;
    variable.right = right;
    variable.op = op;
    variable.items = tensor_zeros(left->items.length);
    variable.grad = tensor_ones(left->items.length);

    va_end(args);
    return variable;
}

Tensor chain_rule_mul(Variable variable) {
    Tensor result = tensor_zeros(variable.grad.length);
    for (int i = 0; i < variable.grad.length; ++i) {
        result.data[i] = variable.items.data[i] * variable.grad.data[i];
    }
    return result;
}

Tensor chain_rule_div_numerator(Variable variable) {
    Tensor result = tensor_zeros(variable.grad.length);
    for (int i = 0; i < variable.grad.length; ++i) {
        result.data[i] = 1 / variable.items.data[i];
    }
    return result;
}

Tensor chain_rule_div_denominator(Variable left, Variable right) {
    assert(left.grad.length == right.grad.length &&
           "Tensor gradients must be of same length");
    size_t n = left.grad.length;
    Tensor result = tensor_zeros(n);
    for (int i = 0; i < n; ++i) {
        float u = left.items.data[i];
        float v = right.items.data[i];
        result.data[i] = (-u * left.grad.data[i]) / (v * v);
    }
    return result;
}

Tensor variable_forward(Variable *root, Graph *visited) {
    if (!visited) {
        Graph graph = graph_empty();
        visited = &graph;
    }

    if (root == NULL)
        return tensor_zeros(0);

    if (graph_is_visited(visited, root))
        return root->items;
    graph_mark_visited(visited, root);

    if (root->op == OP_LEAF || root->left == NULL) {
        return root->items;
    }

#ifndef __fwd_case
#define __fwd_case(operator, func)                                             \
    if (root->op == (operator)) {                                              \
        Tensor left = variable_forward(root->left, visited);                   \
        Tensor right = variable_forward(root->right, visited);                 \
        return (func)(left, right);                                            \
    }
#endif // __fwd_case

    if (root->items.shape.length > 1)
        tensor_reset_shape(&root->items);

    __fwd_case(OP_ADD, tensor_add);
    __fwd_case(OP_SUB, tensor_sub);
    __fwd_case(OP_MUL, tensor_mul);
    if (root->op == (OP_DIV)) {
        Tensor lt = root->left->items;
        Tensor rt = root->right->items;
        tensor_print(lt, {});
        tensor_print(rt, {});
        Tensor left = variable_forward(root->left, visited);
        Tensor right = variable_forward(root->right, visited);
        return (tensor_div)(left, right);
    };
    __fwd_case(OP_DOT, tensor_dot);
    __fwd_case(OP_ACCUM_SUM, tensor_scalar_accumulate);

#ifdef __fwd_case
#undef __fwd_case
#endif // __fwd_case

    return root->items;
}

void variable_backward(Variable *root, Graph *visited) {
    if (!visited) {
        Graph graph = graph_empty();
        visited = &graph;
    }

    if (root == NULL || root->left == NULL)
        return;

    if (graph_is_visited(visited, root))
        return;
    graph_mark_visited(visited, root);

    switch (root->op) {
    case OP_ADD:
    case OP_SUB:
    case OP_ACCUM_SUM: {
        root->left->grad = tensor_ones(root->left->grad.length);
        root->right->grad = tensor_ones(root->right->grad.length);
    } break;
    case OP_DOT:
    case OP_MUL: {
        Tensor left_grad = chain_rule_mul(*root->right);
        Tensor right_grad = chain_rule_mul(*root->left);
        root->left->grad = left_grad;
        root->right->grad = right_grad;
    } break;
    case OP_DIV: {
        Tensor left_grad = chain_rule_div_numerator(*root->right);
        Tensor right_grad =
            chain_rule_div_denominator(*root->left, *root->right);
        root->left->grad = left_grad;
        root->right->grad = right_grad;
    } break;
    default:
        break;
    }

    variable_backward(root->left, visited);
    variable_backward(root->right, visited);
}
