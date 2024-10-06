#include "tensor.h"
#include "variable.h"

void chain_rule_add(struct Variable *root, struct Variable *left, struct Variable *right);
void chain_rule_mul(struct Variable *root, struct Variable *left, struct Variable *right);
void chain_rule_div(struct Variable *root, struct Variable *left, struct Variable *right);
void chain_rule_pow(struct Variable *root, struct Variable *left, struct Variable *right);

