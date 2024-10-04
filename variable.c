#include "variable.h"
#include "tensor.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Variable variable_new(Tensor tensor) {
    Variable variable;
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
        strcmp(op_str, "@") != 0 && strcmp(op_str, "[+]") != 0 &&
        strcmp(op_str, "<+>") != 0 && strcmp(op_str, "<->") != 0 &&
        strcmp(op_str, "<*>") != 0 && strcmp(op_str, "<^>") != 0) {
        fprintf(stderr,
                "Invalid op character. Expected one of {'+', '-', '*', '/', "
                "'@', '[+]', '<+>', '<->', '<*>', '<^>'}, but found %s\n",
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
    else if (strcmp(op_str, "[+]") == 0)
        op = OP_ACCUM_SUM;
    else if (strcmp(op_str, "<+>") == 0)
        op = OP_SCALAR_SUM;
    else if (strcmp(op_str, "<->") == 0)
        op = OP_SCALAR_SUM;
    else if (strcmp(op_str, "<*>") == 0)
        op = OP_SCALAR_MUL;
    else if (strcmp(op_str, "<^>") == 0)
        op = OP_SCALAR_POW;
    else
        op = OP_LEAF;

    Variable variable;
    variable.left = left;
    variable.right = right;
    variable.op = op;
    variable.items = tensor_ones(left->items.length > 0 ? left->items.length
                                                        : right->items.length);
    variable.grad = tensor_ones(left->items.length > 0 ? left->items.length
                                                       : right->items.length);

    va_end(args);

    return variable;
}

Tensor chain_rule_mul(Variable partial, Variable term) {
    Tensor result = tensor_zeros(partial.grad.length);
    for (int i = 0; i < partial.grad.length; ++i) {
        result.data[i] = term.items.data[i] * partial.grad.data[i];
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
    if (left.grad.length == 0) left.grad = tensor_ones(right.grad.length);
    if (right.grad.length == 0) right.grad = tensor_ones(left.grad.length);

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

Tensor chain_rule_pow(Variable base, Variable pow) {
    return tensor_mul(
        pow.grad,
        tensor_scalar_pow(base.items,
                          tensor_scalar_diff(pow.grad, tensor_new_scalar(1))));
}

Tensor chain_rule_base_pow(Variable base, Variable pow) {
    return tensor_scalar_mul(tensor_scalar_pow(base.grad, pow.items),
                             tensor_natural_log(base.grad));
}

Variable variable_forward(Variable *root) {
#define __fwd_case(operator, func)                                             \
    if (root->op == (operator) && root->left->items.length > 0 &&              \
        root->right->items.length > 0) {                                       \
        Variable left = variable_forward(root->left);                          \
        Variable right = variable_forward(root->right);                        \
        if (left.items.length == 0 && right.items.length == 0)                 \
            return variable_new(tensor_zeros(0));                              \
        if (left.items.length == 0)                                            \
            return right;                                                      \
        if (right.items.length == 0)                                           \
            return left;                                                       \
        root->items = (func)(left.items, right.items);                         \
        if (root == NULL)                                                      \
            return variable_new(tensor_zeros(0));                              \
        return *root;                                                          \
    }

    if (root == NULL)
        return variable_new(tensor_zeros(0));
    if (root->left == NULL)
        return *root;
    if (root->right == NULL)
        return *root;

    __fwd_case(OP_ADD, tensor_add);
    __fwd_case(OP_SUB, tensor_sub);
    __fwd_case(OP_MUL, tensor_mul);
    __fwd_case(OP_DIV, tensor_div);
    __fwd_case(OP_DOT, tensor_dot);
    __fwd_case(OP_ACCUM_SUM, tensor_scalar_accumulate);
    __fwd_case(OP_SCALAR_SUM, tensor_scalar_sum);
    __fwd_case(OP_SCALAR_DIFF, tensor_scalar_diff);
    __fwd_case(OP_SCALAR_MUL, tensor_scalar_mul);
    __fwd_case(OP_SCALAR_POW, tensor_scalar_pow);

    return variable_new(tensor_zeros(0));

#undef __fwd_case
}

void variable_backward(Variable *root) {
    if (root == NULL || root->left == NULL || root->right == NULL) {
        return;
    }

    if (root->left->grad.length == 0)
        root->left->grad = tensor_ones(root->left->items.length);
    if (root->right->grad.length == 0)
        root->right->grad = tensor_ones(root->right->items.length);

#define __add_derivative                                                       \
    do {                                                                       \
        root->left->grad = tensor_ones(root->left->grad.length);               \
        root->right->grad = tensor_ones(root->right->grad.length);             \
    } while (0);
#define __sub_derivative                                                       \
    do {                                                                       \
        root->left->grad = tensor_ones(root->left->grad.length);               \
        root->right->grad = tensor_from(root->right->grad.length, -1);         \
    } while (0);
#define __mul_derivative                                                       \
    do {                                                                       \
        root->left->grad = chain_rule_mul(*root, *root->right);                \
        root->right->grad = chain_rule_mul(*root, *root->left);                \
    } while (0);
#define __div_derivative                                                       \
    do {                                                                       \
        Tensor left_grad = chain_rule_div_numerator(*root->right);             \
        Tensor right_grad =                                                    \
            chain_rule_div_denominator(*root->left, *root->right);             \
        root->left->grad = left_grad;                                          \
        root->right->grad = right_grad;                                        \
    } while (0);
#define __pow_derivative                                                       \
    do {                                                                       \
        Tensor left_grad = chain_rule_pow(*root->left, *root->right);          \
        Tensor right_grad = chain_rule_base_pow(*root->left, *root->right);    \
        root->left->grad = left_grad;                                          \
        root->right->grad = right_grad;                                        \
    } while (0);

#define __case(op, fn)                                                         \
    case (op):                                                                 \
        fn break

    switch (root->op) {
        __case(OP_ADD, __add_derivative);
        __case(OP_SCALAR_SUM, __add_derivative);
        __case(OP_ACCUM_SUM, __add_derivative);
        __case(OP_SUB, __sub_derivative);
        __case(OP_SCALAR_DIFF, __sub_derivative);
        __case(OP_MUL, __mul_derivative);
        __case(OP_DOT, __mul_derivative);
        __case(OP_SCALAR_MUL, __mul_derivative);
        __case(OP_DIV, __div_derivative);
        __case(OP_SCALAR_POW, __pow_derivative);
    default:
        break;
    }

    variable_backward(root->left);
    variable_backward(root->right);

#undef __add_derivative
#undef __sub_derivative
#undef __mul_derivative
#undef __div_derivative
#undef __pow_derivative
}

Variable var_copy(Variable variable, bool preserve_tree) {
    return (Variable){.left = preserve_tree ? variable.left : NULL,
                      .right = preserve_tree ? variable.right : NULL,
                      .items = tensor_copy(variable.items),
                      .grad = tensor_copy(variable.grad),
                      .op = variable.op};
}
