#include "variable.h"
#include "chain_rule.h"
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
    variable.grad = tensor_zeros(tensor.length);
    variable.backward = NULL;
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
        strcmp(op_str, "<*>") != 0 && strcmp(op_str, "</>") != 0 &&
        strcmp(op_str, "<^>") != 0 && strcmp(op_str, "**") != 0) {
        fprintf(stderr,
                "Invalid op character. Expected one of {'+', '-', '*', '/', "
                "'@', '[+]', '<+>', '<->', '<*>', '</>', '<^>', '**'}, but "
                "found %s\n",
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
    else if (strcmp(op_str, "</>") == 0)
        op = OP_SCALAR_DIV;
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

#define __fwd_case(operator, func, BACKWARD)                                   \
    if (variable.op == (operator)) {                                           \
        variable.items = (func)(variable.left->items, variable.right->items);  \
        variable.backward = &(BACKWARD);                                       \
        return variable;                                                       \
    }

#define __fwd_case_self(operator, func, BACKWARD)                              \
    if (variable.op == (operator)) {                                           \
        variable.items = (func)(variable.left->items);                         \
        variable.backward = &(BACKWARD);                                       \
        return variable;                                                       \
    }

    __fwd_case(OP_ADD, tensor_add, chain_rule_add);
    __fwd_case(OP_SUB, tensor_sub, chain_rule_add);
    __fwd_case_self(OP_ACCUM_SUM, tensor_sum, chain_rule_add);
    __fwd_case(OP_SCALAR_SUM, tensor_scalar_sum, chain_rule_add);
    __fwd_case(OP_SCALAR_DIFF, tensor_scalar_diff, chain_rule_add);
    __fwd_case(OP_MUL, tensor_mul, chain_rule_mul);
    __fwd_case(OP_DOT, tensor_dot, chain_rule_mul);
    __fwd_case(OP_SCALAR_MUL, tensor_scalar_mul, chain_rule_mul);
    __fwd_case(OP_DIV, tensor_div, chain_rule_div);
    __fwd_case(OP_SCALAR_DIV, tensor_scalar_mul, chain_rule_div);
    __fwd_case(OP_SCALAR_POW, tensor_scalar_pow, chain_rule_pow);
    __fwd_case_self(OP_SCALAR_SQ, tensor_scalar_sq, chain_rule_pow);

#undef __fwd_case
#undef __fwd_case_self

    return variable;
}

void variable_backward(Variable *root) {
    if (DUMB_NULL_CHECK(root) || DUMB_NULL_CHECK(root->backward)) {
        return;
    }

    if (!DUMB_NULL_CHECK(root->left) && !DUMB_NULL_CHECK(root->right)) {
        (*root->backward)(root, root->left, root->right);
    }

    if (!DUMB_NULL_CHECK(root->left))
        variable_backward(root->left);
    if (!DUMB_NULL_CHECK(root->right))
        variable_backward(root->right);
}

Variable var_copy(Variable variable, bool preserve_tree) {
    return (Variable){.left = preserve_tree ? variable.left : NULL,
                      .right = preserve_tree ? variable.right : NULL,
                      .items = tensor_copy(variable.items),
                      .grad = tensor_copy(variable.grad),
                      .op = variable.op};
}
