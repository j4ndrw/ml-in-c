#include <assert.h>
#include <stdarg.h>

#include "variable.h"

int main() {
    VAR_INIT(a, 0.0f, 1.0f, 2.0f, 3.0f);
    VAR_INIT(b, 2.0f, 4.0f, 6.0f, 8.0f);
    VAR(c, OP(&a, *, &b));
    Tensor result = FORWARD(&c);
    BACKWARD(&c);
    BACKWARD(&c);
    BACKWARD(&c);
    return 0;
}
