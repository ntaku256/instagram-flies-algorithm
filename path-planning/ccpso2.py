import numpy as np
from copy import deepcopy

def CCPSO2(problem, *,
           S=(2,5,10,50,100),
           pop_size=30,
           max_iter=100,
           p_cauchy=0.5,
           var_min=None,
           var_max=None,
           callback=None,
           seed=None):
    """
    CCPSO2 implementation (practical version inspired by Yao & Li).
    Returns: gbest (dict), final_pop (list of particle dicts)
    Notes:
      - cost_function must return tuple (cost, details)
      - S: iterable of candidate group sizes (integers)
    """
    if seed is not None:
        np.random.seed(seed)

    cost_function = problem['cost_function']
    call_count = {'count': 0}

    def counted_cost_function(x):
        call_count['count'] += 1
        return cost_function(x)
    
    n = problem['num_var']
    # bounds (support scalar or vector)
    var_min = problem['var_min'] if var_min is None else var_min
    var_max = problem['var_max'] if var_max is None else var_max
    if np.isscalar(var_min):
        var_min = np.full(n, var_min, dtype=float)
    else:
        var_min = np.array(var_min, dtype=float)
    if np.isscalar(var_max):
        var_max = np.full(n, var_max, dtype=float)
    else:
        var_max = np.array(var_max, dtype=float)

    # Initialize population matrices X (current) and Y (personal bests)
    X = np.random.uniform(var_min, var_max, size=(pop_size, n))
    X_costs = np.full(pop_size, np.inf)
    X_details = [None]*pop_size

    # Evaluate initial X
    for i in range(pop_size):
        X_costs[i], X_details[i] = counted_cost_function(X[i].copy())

    Y = X.copy()
    Y_costs = X_costs.copy()
    Y_details = deepcopy(X_details)

    # Global best (context vector y_hat)
    best_idx = int(np.argmin(Y_costs))
    y_hat = Y[best_idx].copy()
    y_hat_cost = float(Y_costs[best_idx])
    y_hat_details = deepcopy(Y_details[best_idx])

    gbest = {'position': y_hat.copy(), 'cost': y_hat_cost, 'details': y_hat_details}

    # helper: permute indices and partition into groups of size s
    def make_groups(s):
        perm = np.random.permutation(n)
        groups = []
        i = 0
        while i < n:
            j = min(i + s, n)
            groups.append(perm[i:j])
            i = j
        return groups, perm

    # helper: local best (ring neighbourhood) within a sub-swarm (using Y_costs)
    def local_best_index_in_subswarm(sub_cols, particle_index):
        # neighbourhood: left, self, right (wrap) as in paper
        left = (particle_index - 1) % pop_size
        right = (particle_index + 1) % pop_size
        candidates = [left, particle_index, right]
        best = candidates[0]
        for c in candidates:
            if Y_costs[c] < Y_costs[best]:
                best = c
        return best

    # main loop (each iteration == one coevolution cycle)
    last_yhat_cost = y_hat_cost
    for it in range(1, max_iter + 1):
        # if no improvement in last cycle, choose a new group size s uniformly from S
        if it == 1 or y_hat_cost < last_yhat_cost:
            # keep same s if improvement, else choose new s
            s_current = None if it == 1 else s_current
        else:
            s_current = int(np.random.choice(S))

        # if first iteration and s_current not set, pick one
        if it == 1 and s_current is None:
            s_current = int(np.random.choice(S))

        last_yhat_cost = y_hat_cost

        # random grouping: permute indices and split into groups of size s_current
        groups, perm = make_groups(s_current)
        K = len(groups)

        # --- Step 1: For each sub-swarm, check personal-best updates using context y_hat ---
        # Implementation of the paper's b(j, .) using context y_hat:
        for j, cols in enumerate(groups):
            cols = np.array(cols)
            # for each particle in the sub-swarm
            for i in range(pop_size):
                # candidate formed by replacing j-th block in y_hat with current particle's subvector
                cand = y_hat.copy()
                cand[cols] = X[i, cols]
                cand_cost, cand_det = counted_cost_function(cand)
                # candidate for pbest (replace j-th block with personal-best's subvector)
                cand_pbest = y_hat.copy()
                cand_pbest[cols] = Y[i, cols]
                cand_pbest_cost, _ = counted_cost_function(cand_pbest)

                # if candidate better, update the personal best subvector (columns)
                if cand_cost < cand_pbest_cost:
                    # update Y's columns from X's columns for this particle
                    Y[i, cols] = X[i, cols].copy()
                    # recompute the full Y_cost for this particle: evaluate full Y[i]
                    Y_costs[i], Y_details[i] = counted_cost_function(Y[i].copy())

            # after processing all particles in this sub-swarm, update sub-swarm global best
            # (we won't keep a separate sub_gbest array; instead we will use Y_costs to find best per block)
            # If replacing block in y_hat with the best pbest subvector improves y_hat, update y_hat
            # find best particle for this block by Y_costs
            best_particle_idx = int(np.argmin(Y_costs))
            cand2 = y_hat.copy()
            cand2[cols] = Y[best_particle_idx, cols]
            cand2_cost, cand2_det = counted_cost_function(cand2)
            if cand2_cost < y_hat_cost:
                y_hat = cand2.copy()
                y_hat_cost = float(cand2_cost)
                y_hat_details = cand2_det

        # --- Step 2: compute local bests (lbest ring) in each sub-swarm and perform CGPSO updates per sub-swarm ---
        for j, cols in enumerate(groups):
            cols = np.array(cols)
            # for each particle i compute local-best index in this sub-swarm (using Y_costs)
            local_best_indices = [local_best_index_in_subswarm(cols, i) for i in range(pop_size)]

            # now update positions using CGPSO formula (Eq.4)
            for i in range(pop_size):
                yi_sub = Y[i, cols].copy()  # personal best subvector
                yhatprime_idx = local_best_indices[i]
                yhatprime_sub = Y[yhatprime_idx, cols].copy()  # local neighbourhood best subvector

                # step-by-step per-dimension
                diff = np.abs(yi_sub - yhatprime_sub)
                # to avoid zero-scale, set minimal scale
                eps = 1e-12
                scale = np.maximum(diff, eps)

                # decide per-dimension whether to use Cauchy or Gaussian
                r = np.random.rand(len(cols))
                use_cauchy = r <= p_cauchy

                # Cauchy sample around yi_sub
                if np.any(use_cauchy):
                    # standard cauchy has heavy tails: use scale * cauchy_sample
                    cauch = np.random.standard_cauchy(size=use_cauchy.sum())
                    # select indices
                    idx = np.where(use_cauchy)[0]
                    yi_sub[idx] = yi_sub[idx] + cauch * scale[idx]

                # Gaussian sample around yhatprime_sub otherwise
                if np.any(~use_cauchy):
                    idxg = np.where(~use_cauchy)[0]
                    gauss = np.random.randn(idxg.size)
                    yi_sub[idxg] = yhatprime_sub[idxg] + gauss * scale[idxg]

                # write back updated subvector into X (current position)
                X[i, cols] = np.clip(yi_sub, var_min[cols], var_max[cols])

        # --- Step 3: after all sub-swarms updated, evaluate X (full vectors) and update personal bests & global best ---
        for i in range(pop_size):
            X_costs[i], X_details[i] = counted_cost_function(X[i].copy())
            # if this full current X is better than Y (personal best), update full Y
            if X_costs[i] < Y_costs[i]:
                Y[i] = X[i].copy()
                Y_costs[i] = X_costs[i]
                Y_details[i] = X_details[i]

        # update global best y_hat if any Y is better
        best_idx = int(np.argmin(Y_costs))
        if Y_costs[best_idx] < y_hat_cost:
            y_hat = Y[best_idx].copy()
            y_hat_cost = float(Y_costs[best_idx])
            y_hat_details = deepcopy(Y_details[best_idx])

        # optional callback / progress
        gbest = {'position': y_hat.copy(), 'cost': y_hat_cost, 'details': y_hat_details}
        if callable(callback):
            callback({'it': it, 'gbest': deepcopy(gbest), 'X': X.copy(), 'X_costs': X_costs.copy()})

        # (loop continues)
        print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))
        print("Total cost_function calls:", call_count['count'])

    # construct final population info (complete particles)
    final_pop = []
    for i in range(pop_size):
        p = {
            'position': X[i].copy(),
            'cost': float(X_costs[i]),
            'details': X_details[i],
            'best': {
                'position': Y[i].copy(),
                'cost': float(Y_costs[i]),
                'details': Y_details[i],
            },
        }
        final_pop.append(p)

    return gbest, final_pop
