/* Per-branch events — alloc/free for recount_branch_stats_t.
 *
 * The compute path (recount_branch_stats_batch) lives in recount_gradient.c
 * because it reuses the per-family inside/outside pipeline; this file only
 * owns the struct lifecycle. Validated bit-for-bit against Java
 * count.model.Posteriors.getNodeMean / getNodePosteriors on the focal
 * arc269 + Williams2017 datasets — see VALIDATION.md. */
#include <stdlib.h>
#include <string.h>
#include "recount_native.h"

int recount_branch_stats_alloc(recount_branch_stats_t *bs, int32_t num_nodes) {
    bs->num_nodes = num_nodes;
    size_t n = (size_t)num_nodes;
    /* Allocate 5 contiguous doubles arrays: copies_node, copies_edge,
     * gain_events, loss_events, families_present. */
    bs->copies_node = (double *)calloc(5 * n, sizeof(double));
    if (!bs->copies_node) return RECOUNT_ENOMEM;
    bs->copies_edge = bs->copies_node + n;
    bs->gain_events = bs->copies_node + 2 * n;
    bs->loss_events = bs->copies_node + 3 * n;
    bs->families_present = bs->copies_node + 4 * n;
    bs->num_families_active = (int32_t *)calloc(n, sizeof(int32_t));
    if (!bs->num_families_active) {
        free(bs->copies_node); bs->copies_node = NULL;
        return RECOUNT_ENOMEM;
    }
    return RECOUNT_OK;
}

void recount_branch_stats_free(recount_branch_stats_t *bs) {
    free(bs->copies_node);
    free(bs->num_families_active);
    memset(bs, 0, sizeof(*bs));
}

/* recount_branch_stats_batch is implemented in recount_gradient.c
 * (shares the per-family forward+outside pipeline with the gradient). */
