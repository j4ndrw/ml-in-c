#!/bin/bash

set -xe

function find_all_c_files()
{
    find . -type f -name "*.c" | xargs echo
}

function compile()
{
    xargs -I % bash -c "$1"
}

# I disabled the stack protector because I am absolutely not going to refactor
# this fucking mess of a code base to get rid of buffer overflows...
find_all_c_files | compile "gcc -std=c99 -ggdb -o main % -lm -fno-stack-protector"
