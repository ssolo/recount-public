/* recount_tree.c — derive a CSR children list from a parent[] array.
 *
 * recount's tree convention (inherited from Csurös' Count) indexes
 * leaves first and requires parent[v] > v for every non-root. This
 * file builds the inverse adjacency (first_child[v], child_list[]) once
 * per session so post-order and pre-order traversals run in linear
 * time.
 */
#include "recount_native.h"
#include <string.h>

/* Build CSR children list from a parent[] array.
 * Caller-owned: first_child[num_nodes+1], child_list[num_nodes-1].
 *
 * After this:
 *   children of node v are child_list[first_child[v] .. first_child[v+1])
 */
void recount_tree_build_csr(
    int32_t num_nodes, int32_t root, const int32_t *parent,
    int32_t *out_first_child, int32_t *out_child_list)
{
    /* count children per node */
    memset(out_first_child, 0, (size_t)(num_nodes + 1) * sizeof(int32_t));
    for (int32_t v = 0; v < num_nodes; v++) {
        if (v == root) continue;
        int32_t p = parent[v];
        out_first_child[p + 1] += 1;  /* offset by 1 for prefix-sum below */
    }
    /* prefix-sum to make first_child indices */
    for (int32_t v = 0; v < num_nodes; v++) {
        out_first_child[v + 1] += out_first_child[v];
    }
    /* scatter children */
    int32_t *cursor = (int32_t *)__builtin_alloca((size_t)num_nodes * sizeof(int32_t));
    memset(cursor, 0, (size_t)num_nodes * sizeof(int32_t));
    for (int32_t v = 0; v < num_nodes; v++) {
        if (v == root) continue;
        int32_t p = parent[v];
        int32_t pos = out_first_child[p] + cursor[p];
        out_child_list[pos] = v;
        cursor[p]++;
    }
}
