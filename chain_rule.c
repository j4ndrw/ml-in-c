#include "tensor.h"
#include "variable.h"

#define ACC(grad, expr) (grad) = ADD((grad), (expr))

void chain_rule_add(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad, root->grad);
    ACC(right->grad, root->grad);
}

void chain_rule_sub(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad, root->grad);
    ACC(right->grad, SMUL(root->grad, SNEW(-1)));
}

void chain_rule_mul(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad, MUL(right->items, root->grad));
    ACC(right->grad, MUL(left->items, root->grad));
}

void chain_rule_div(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad, MUL(root->grad, SPOW(right->items, SNEW(-1))));
    ACC(right->grad,
        MUL(root->grad, SMUL(DIV(left->items, SSQ(right->items)), SNEW(-1))));
}

void chain_rule_pow(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad,
        MUL(root->grad,
            MUL(right->items, POW(left->items, SDIFF(right->items, SNEW(1))))));
    ACC(right->grad,
        MUL(root->grad, MUL(POW(left->items, right->items), LN(left->items))));
}

void chain_rule_exp(struct Variable *root, struct Variable *left,
                    struct Variable *right) {
    ACC(left->grad, EXP(left->items));
    ACC(right->grad, SNEW(0));
}
