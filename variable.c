#include "variable.h"
#include "tensor.h"
#include "utils.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Variable variable_new(Tensor tensor) {
    Variable variable;
    variable.left = (Variable *)NULL;
    variable.right = (Variable *)NULL;
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
        strcmp(op_str, "<*>") != 0 && strcmp(op_str, "<^>") != 0 &&
        strcmp(op_str, "**") != 0) {
        fprintf(stderr,
                "Invalid op character. Expected one of {'+', '-', '*', '/', "
                "'@', '[+]', '<+>', '<->', '<*>', '<^>', '**'}, but found %s\n",
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
    else if (strcmp(op_str, "**") == 0)
        op = OP_SCALAR_SQ;
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

Variable variable_forward(Variable *root) {
    if (DUMB_NULL_CHECK(root))
        return variable_new(tensor_empty(1));

#define __fwd_case(operator, func)                                             \
    if (root->op == (operator)) {                                              \
        Variable left = variable_forward(root->left);                          \
        Variable right = variable_forward(root->right);                        \
        if (left.items.length == 0 || right.items.length == 0)                 \
            return variable_new(tensor_empty(1));                              \
        root->items = (func)(left.items, right.items);                         \
        return *root;                                                          \
    }

#define __fwd_case_self(operator, func)                                        \
    if (root->op == (operator)) {                                              \
        Variable left = variable_forward(root->left);                          \
        if (left.items.length == 0)                                            \
            return variable_new(tensor_empty(1));                              \
        root->items = (func)(left.items);                                      \
        return *root;                                                          \
    }

    __fwd_case(OP_ADD, tensor_add);
    __fwd_case(OP_SUB, tensor_sub);
    __fwd_case(OP_MUL, tensor_mul);
    __fwd_case(OP_DIV, tensor_div);
    __fwd_case(OP_DOT, tensor_dot);
    __fwd_case_self(OP_ACCUM_SUM, tensor_sum);
    __fwd_case(OP_SCALAR_SUM, tensor_scalar_sum);
    __fwd_case(OP_SCALAR_DIFF, tensor_scalar_diff);
    __fwd_case(OP_SCALAR_MUL, tensor_scalar_mul);
    __fwd_case(OP_SCALAR_POW, tensor_scalar_pow);
    __fwd_case_self(OP_SCALAR_SQ, tensor_scalar_sq);

#undef __fwd_case

    if (DUMB_NULL_CHECK(root))
        return variable_new(tensor_empty(1));
    return *root;
}

void variable_backward(Variable *root) {
    if (DUMB_NULL_CHECK(root) ||
        (DUMB_NULL_CHECK(root->left) && DUMB_NULL_CHECK(root->right)))
        return;
    if (DUMB_NULL_CHECK(root->left) && !DUMB_NULL_CHECK(root->right))
        return variable_backward(root->right);
    if (DUMB_NULL_CHECK(root->right) && !DUMB_NULL_CHECK(root->left))
        return variable_backward(root->left);

    Op op = root->op;
    Tensor grad = root->grad;
    Tensor left_grad = root->left->grad;
    Tensor left_items = root->left->items;
    Tensor right_grad = root->right->grad;
    Tensor right_items = root->right->items;

#define __add_derivative                                                       \
    chain_rule_add(new_left_grad, grad);                                       \
    chain_rule_add(new_right_grad, grad);                                      \
    left_grad = new_left_grad;                                                 \
    right_grad = new_right_grad;

#define __mul_derivative                                                       \
    chain_rule_mul(new_left_grad, grad, right_items);                          \
    chain_rule_mul(new_right_grad, grad, left_items);                          \
    left_grad = new_left_grad;                                                 \
    right_grad = new_right_grad;

#define __div_derivative                                                       \
    chain_rule_div_numerator(new_left_grad, right_grad, right_items);          \
    chain_rule_div_denominator(new_right_grad, left_grad, left_items,          \
                               right_items);                                   \
    left_grad = new_left_grad;                                                 \
    right_grad = new_right_grad;

#define __pow_derivative                                                       \
    chain_rule_pow(new_left_grad, right_grad, left_items, right_items);        \
    chain_rule_base_pow(new_right_grad, left_grad, left_items, right_items);   \
    left_grad = new_left_grad;                                                 \
    right_grad = new_right_grad;

#define __case(OP, FN)                                                         \
    do {                                                                       \
        if (op == OP) {                                                        \
            FN;                                                                \
            for (size_t i = 0; i < root->left->grad.length; ++i) {             \
                root->left->grad.data[i] = left_grad.data[i];                       \
            }                                                                  \
            for (size_t i = 0; i < root->right->grad.length; ++i) {             \
                root->right->grad.data[i] = right_grad.data[i];                       \
            }                                                                  \
            variable_backward(root->left);                                     \
            variable_backward(root->right);                                    \
            return;                                                            \
        }                                                                      \
    } while (0)

#define __default                                                              \
    do {                                                                       \
        variable_backward(root->left);                                         \
        variable_backward(root->right);                                        \
        return;                                                                \
    } while (0)

    __case(OP_ADD, __add_derivative);
    __case(OP_SCALAR_SUM, __add_derivative);
    __case(OP_ACCUM_SUM, __add_derivative);
    __case(OP_SUB, __add_derivative);
    __case(OP_SCALAR_DIFF, __add_derivative);
    __case(OP_MUL, __mul_derivative);
    __case(OP_DOT, __mul_derivative);
    __case(OP_SCALAR_MUL, __mul_derivative);
    __case(OP_DIV, __div_derivative);
    __case(OP_SCALAR_POW, __pow_derivative);
    __case(OP_SCALAR_SQ, __pow_derivative);
    __default;

#undef __add_derivative
#undef __sub_derivative
#undef __mul_derivative
#undef __div_derivative
#undef __pow_derivative
#undef __case
#undef __default
}

Variable var_copy(Variable variable, bool preserve_tree) {
    return (Variable){.left = preserve_tree ? variable.left : NULL,
                      .right = preserve_tree ? variable.right : NULL,
                      .items = tensor_copy(variable.items),
                      .grad = tensor_copy(variable.grad),
                      .op = variable.op};
}
