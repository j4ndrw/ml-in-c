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

find_all_c_files | compile "gcc -std=c99 -ggdb -Wall -Wextra -o main % -lm"
