/* recount_parallel.h — single-macro parallel-for portability layer.
 *
 * On Apple platforms we use libdispatch (Grand Central Dispatch); on
 * Linux / other Unixes we fall back to OpenMP. The Apple path is what
 * production M-series builds use; the OpenMP path is for portability to
 * Intel Macs and Linux (CI, server deployments).
 *
 * Usage at each call site:
 *
 *   RECOUNT_PARALLEL_APPLY(nworkers, {
 *       // worker_id is the iteration index, [0, nworkers)
 *       // shared variables in the surrounding scope are captured
 *       // (block-capture on Apple, OpenMP default-shared elsewhere).
 *       grad_worker_t *wk = &workers[worker_id];
 *       for (;;) {
 *           int32_t f = __sync_fetch_and_add(&next_idx, 1);
 *           if (f >= F) break;
 *           ...
 *       }
 *   });
 *
 * Numerical results are identical on both paths (the per-iteration body
 * is the same; only the worker dispatch differs).
 */
#ifndef RECOUNT_PARALLEL_H
#define RECOUNT_PARALLEL_H

#include <stddef.h>

#ifdef __APPLE__
  #include <dispatch/dispatch.h>
  #define RECOUNT_PARALLEL_APPLY(n_workers, BODY)                                       \
      do {                                                                              \
          dispatch_queue_t _recount_q =                                                 \
              dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);                   \
          dispatch_apply((size_t)(n_workers), _recount_q,                               \
                         ^(size_t worker_id) BODY );                                    \
      } while (0)
#else
  /* Non-Apple: prefer OpenMP if available; otherwise fall back to a serial loop. */
  #ifdef _OPENMP
    #include <omp.h>
    #define RECOUNT_PARALLEL_APPLY(n_workers, BODY)                                     \
        do {                                                                            \
            _Pragma("omp parallel for schedule(static, 1)")                             \
            for (size_t worker_id = 0; worker_id < (size_t)(n_workers); worker_id++)    \
                BODY ;                                                                  \
        } while (0)
  #else
    #define RECOUNT_PARALLEL_APPLY(n_workers, BODY)                                     \
        do {                                                                            \
            for (size_t worker_id = 0; worker_id < (size_t)(n_workers); worker_id++)    \
                BODY ;                                                                  \
        } while (0)
  #endif
#endif

#endif /* RECOUNT_PARALLEL_H */
