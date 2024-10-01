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

find_all_c_files | compile "cc -std=c99 -ggdb -o main %"
