#include <assert.h>
#include <stdarg.h>
#include <time.h>

#include "variable.h"

int main() {
    srand(time(NULL));
    size_t shape[] = {2, 2};

    var_rand(a, 4, {2, 2});
    var_print(a, items, {2, 2});
    return 0;
}
