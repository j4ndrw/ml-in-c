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
    variable.grad = tensor_ones(left->items.length);

#ifndef __fwd_case
#define __fwd_case(operator, func)                                             \
    if (variable.op == (operator) && variable.left->items.length > 0 &&        \
        variable.right->items.length > 0) {                                    \
        variable.items = (func)(variable.left->items, variable.right->items);  \
        return variable;                                                       \
    }
#endif // __fwd_case

    va_end(args);

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

void variable_backward(Variable *root) {
    if (root == NULL || root->left == NULL || root->right == NULL) {
        return;
    }

    tensor_reset_shape(&root->left->items);
    tensor_reset_shape(&root->right->items);
    tensor_reset_shape(&root->left->grad);
    tensor_reset_shape(&root->right->grad);

    if (root->left->grad.data == NULL || root->left->items.data == NULL ||
        root->left->items.length == 0 || root->left->items.shape.length == 0 ||
        root->left->items.shape.data == NULL) {
        return;
    }

    if (root->right->grad.data == NULL || root->right->items.data == NULL ||
        root->right->items.length == 0 ||
        root->right->items.shape.length == 0 ||
        root->right->items.shape.data == NULL) {
        return;
    }
    Variable left = *root->left;
    Variable right = *root->right;

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
        Tensor left_grad = chain_rule_mul(*root->right);                       \
        Tensor right_grad = chain_rule_mul(*root->left);                       \
        root->left->grad = left_grad;                                          \
        root->right->grad = right_grad;                                        \
    } while (0);
#define __div_derivative                                                       \
    do {                                                                       \
        root->left->grad = chain_rule_div_numerator(right);                    \
        root->right->grad = chain_rule_div_denominator(left, right);           \
    } while (0);
#define __pow_derivative                                                       \
    do {                                                                       \
        root->left->grad = chain_rule_pow(left, right);                        \
        root->right->grad = chain_rule_base_pow(left, right);                  \
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
