/* recount_factln.c — log-factorial and log-rising-factorial caches.
 *
 * Tables hold log(n!) and log(Γ(κ+n)/Γ(κ)) for n in [0, max_n]. Used
 * inside the per-family edge step (Poisson and Pólya gain PMFs) and the
 * destructive sibling combine's multinomial weights. Small (a few KB
 * each) and built once per dataset / per Pólya node.
 */
#include "recount_native.h"
#include "recount_internal.h"

#include <stdlib.h>
#include <math.h>

int recount_factln_init(recount_factln_t *f, int max_n) {
    if (max_n < 0) return RECOUNT_EINVAL;
    f->max_n = max_n;
    f->table = (double *)malloc((size_t)(max_n + 1) * sizeof(double));
    if (!f->table) return RECOUNT_ENOMEM;
    for (int i = 0; i <= max_n; i++) {
        f->table[i] = lgamma((double)(i + 1));
    }
    return RECOUNT_OK;
}

void recount_factln_free(recount_factln_t *f) {
    free(f->table);
    f->table = NULL;
    f->max_n = -1;
}

double recount_factln_get(const recount_factln_t *f, int n) {
    if (n <= f->max_n) return f->table[n];
    /* Fallback: lgamma directly. */
    return lgamma((double)(n + 1));
}

int recount_rfactln_init(recount_rfactln_t *rf, double kappa, int max_n) {
    if (max_n < 0) return RECOUNT_EINVAL;
    rf->max_n = max_n;
    rf->kappa = kappa;
    rf->table = (double *)malloc((size_t)(max_n + 1) * sizeof(double));
    if (!rf->table) return RECOUNT_ENOMEM;
    /* table[n] = lgamma(kappa + n) - lgamma(kappa) */
    rf->table[0] = 0.0;
    for (int i = 1; i <= max_n; i++) {
        rf->table[i] = rf->table[i - 1] + log(kappa + (double)(i - 1));
    }
    return RECOUNT_OK;
}

void recount_rfactln_free(recount_rfactln_t *rf) {
    free(rf->table);
    rf->table = NULL;
    rf->max_n = -1;
}

double recount_rfactln_get(const recount_rfactln_t *rf, int n) {
    if (n >= 0 && n <= rf->max_n) return rf->table[n];
    /* Fallback */
    return lgamma(rf->kappa + (double)n) - lgamma(rf->kappa);
}
