/* Threaded batch via the RECOUNT_PARALLEL_APPLY portability macro
 * (libdispatch on Apple, OpenMP on Linux/Intel). */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "recount_native.h"
#include "recount_internal.h"
#include "recount_parallel.h"

extern int recount_forward_family_with_scratch(
    const recount_tree_t *tree, const recount_survival_t *sp,
    const int32_t *profile,
    const recount_factln_t *fact, const recount_rfactln_t * const *rfacts,
    int32_t *widths, double **C_slots, double **K_slots,
    double *bigpool, size_t bigpool_n, double *out_ll);

static int build_rfacts(const recount_survival_t *sp, int max_n,
                        recount_rfactln_t ***out_rfacts)
{
    int n = sp->num_nodes;
    recount_rfactln_t **rfacts = (recount_rfactln_t **)calloc((size_t)n, sizeof(*rfacts));
    if (!rfacts) return RECOUNT_ENOMEM;
    for (int v = 0; v < n; v++) {
        if (sp->is_polya[v] && sp->gain[v] > 0.0) {
            rfacts[v] = (recount_rfactln_t *)malloc(sizeof(recount_rfactln_t));
            if (!rfacts[v] || recount_rfactln_init(rfacts[v], sp->gain[v], max_n) != RECOUNT_OK) {
                for (int u = 0; u <= v; u++) {
                    if (rfacts[u]) { recount_rfactln_free(rfacts[u]); free(rfacts[u]); }
                }
                free(rfacts);
                return RECOUNT_ENOMEM;
            }
        }
    }
    *out_rfacts = rfacts;
    return RECOUNT_OK;
}

int recount_forward_batch(
    const recount_tree_t *tree,
    const recount_survival_t *sp,
    const int32_t *profiles, int32_t F,
    int32_t num_threads,
    double *out_lls)
{
    int n = tree->num_nodes;
    int num_leaves = tree->num_leaves;

    /* Determine max profile sum across the batch (defines factorial caches). */
    int max_sum = 0, max_count = 0;
    for (int32_t f = 0; f < F; f++) {
        int s = 0;
        const int32_t *prof = profiles + (size_t)f * num_leaves;
        for (int32_t v = 0; v < num_leaves; v++) {
            int32_t c = prof[v];
            if (c > 0) {
                s += c;
                if (c > max_count) max_count = c;
            }
        }
        if (s > max_sum) max_sum = s;
    }
    int max_n = (max_count > max_sum ? max_count : max_sum) + 8;

    /* Shared factorial caches (read-only across threads). */
    recount_factln_t fact;
    if (recount_factln_init(&fact, max_n) != RECOUNT_OK) return RECOUNT_ENOMEM;
    recount_rfactln_t **rfacts = NULL;
    int rc = build_rfacts(sp, max_n, &rfacts);
    if (rc != RECOUNT_OK) { recount_factln_free(&fact); return rc; }

    /* Per-thread scratchpads: one slab per worker. */
    size_t per_scratch = recount_forward_scratch_size(n, max_sum + 1);
    int nworkers;
    if (num_threads > 0) {
        nworkers = num_threads;
    } else {
        nworkers = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nworkers < 1) nworkers = 1;
        if (nworkers > 32) nworkers = 32;
    }
    /* Atomic counter for work-stealing across families (declared before any
     * goto). __block is needed on Apple so the dispatch closure can write to
     * it; plain shared on the OpenMP path. */
#ifdef __APPLE__
    __block int32_t next_idx = 0;
#else
    int32_t next_idx = 0;
#endif

    uint8_t **scratches = (uint8_t **)calloc((size_t)nworkers, sizeof(uint8_t *));
    if (!scratches) { rc = RECOUNT_ENOMEM; goto cleanup; }
    for (int w = 0; w < nworkers; w++) {
        scratches[w] = (uint8_t *)malloc(per_scratch);
        if (!scratches[w]) { rc = RECOUNT_ENOMEM; goto cleanup_scratches; }
    }

    RECOUNT_PARALLEL_APPLY(nworkers, {
        uint8_t *scratch = scratches[worker_id];
        int32_t *widths_buf = (int32_t *)scratch;
        size_t off = (size_t)n * sizeof(int32_t);
        double **C_slots = (double **)(scratch + off);
        off += (size_t)n * sizeof(double *);
        double **K_slots = (double **)(scratch + off);
        off += (size_t)n * sizeof(double *);
        off = (off + 15) & ~(size_t)15;
        double *bigpool = (double *)(scratch + off);
        size_t bigpool_n = (per_scratch - off) / sizeof(double);

        for (;;) {
            int32_t f = __sync_fetch_and_add(&next_idx, 1);
            if (f >= F) break;
            double ll = 0.0;
            int rc1 = recount_forward_family_with_scratch(
                tree, sp, profiles + (size_t)f * num_leaves,
                &fact, (const recount_rfactln_t * const *)rfacts,
                widths_buf, C_slots, K_slots, bigpool, bigpool_n, &ll);
            out_lls[f] = (rc1 == RECOUNT_OK) ? ll : NAN;
        }
    });

cleanup_scratches:
    if (scratches) {
        for (int w = 0; w < nworkers; w++) free(scratches[w]);
        free(scratches);
    }
cleanup:
    if (rfacts) {
        for (int v = 0; v < n; v++) {
            if (rfacts[v]) { recount_rfactln_free(rfacts[v]); free(rfacts[v]); }
        }
        free(rfacts);
    }
    recount_factln_free(&fact);
    return rc;
}
