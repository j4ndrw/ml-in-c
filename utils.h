#pragma once

#define ARR_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define DUMB_NULL_CHECK(ptr)                                                   \
    (((void *)(ptr) == NULL) || ((void *)(ptr) == (void *)0x1))

double randf64();
