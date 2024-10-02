#pragma once

#include <stddef.h>
#include <stdbool.h>

#define GRAPH_INIT_CAP 1024

typedef struct {
    struct Variable **nodes;
    size_t length;
    size_t capacity;
} Graph;


Graph graph_empty();
bool graph_is_visited(Graph *graph, struct Variable *node);
void graph_mark_visited(Graph *graph, struct Variable *node);
