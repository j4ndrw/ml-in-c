#include <assert.h>
#include <stdarg.h>

#include "variable.h"

int main() {
    VAR_INIT(a, {2.0f});
    VAR_INIT(b, {3.0f});
    VAR(c, OP(&a, *, &b));
    Tensor result = FORWARD(&c);
    BACKWARD(&c);
    BACKWARD(&c);
    BACKWARD(&c);
    return 0;
}
