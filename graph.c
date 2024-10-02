#include <stdbool.h>
#include <stdlib.h>

#include "graph.h"
#include "variable.h"

Graph graph_empty() {
    return (Graph){
        .nodes =
            (Variable **)malloc(GRAPH_INIT_CAP * sizeof(Variable *)),
        .length = 0,
        .capacity = GRAPH_INIT_CAP,
    };
}

bool graph_is_visited(Graph *graph, Variable *node) {
    for (size_t i = 0; i < graph->length; i++) {
        if (graph->nodes[i] == node) {
            return true;
        }
    }
    return false;
}

void graph_mark_visited(Graph *graph, Variable *node) {
    if (graph->length >= graph->capacity) {
        graph->capacity = graph->length * 2;
        Variable **new_nodes =
            realloc(graph->nodes, graph->capacity * sizeof(Variable *));
        graph->nodes = new_nodes;
    }
    graph->nodes[graph->length++] = node;
}
