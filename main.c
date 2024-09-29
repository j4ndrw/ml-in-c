#include <assert.h>
#include <stdarg.h>

#include "variable.h"

int main() {
    var_init(a, 8, {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    var_init(b, 8, {2.0f, 4.0f, 6.0f, 8.0f, 2.0f, 4.0f, 6.0f, 8.0f});
    var_expr(c, op(&a, *, &b));
    Tensor result = forward(&c);
    backward(&c);
    backward(&c);
    backward(&c);
    variable_print(a, grad, 2, 2, 2);
    return 0;
}
