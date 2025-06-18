import numpy as np

def IA_V2_Algorithm(problem, **kwargs):
    max_iter = kwargs.get("max_iter", 100)
    pop_size = kwargs.get("pop_size", 100)

    cost_function = problem["cost_function"]
    num_var = problem["num_var"]
    var_min = problem["var_min"]
    var_max = problem["var_max"]
    callback = kwargs.get("callback", None)

    # 初期化
    particles_position = np.random.uniform(var_min, var_max, (pop_size, num_var))
    particles_velocity = np.zeros_like(particles_position)
    particles_cost = np.zeros(pop_size)
    particles_best_position = np.copy(particles_position)
    particles_best_cost = np.full(pop_size, np.inf)
    details_list = [None] * pop_size

    for i in range(pop_size):
        cost, details = cost_function(particles_position[i])
        particles_cost[i] = cost
        details_list[i] = details
        particles_best_cost[i] = cost
        particles_best_position[i] = particles_position[i]

    for it in range(max_iter):
        for i in range(pop_size):
            # 重心（平均）方向へ更新（Instabae風）
            center = np.mean(particles_best_position, axis=0)
            velocity = (center - particles_position[i]) * np.random.rand()
            particles_velocity[i] = 0.5 * particles_velocity[i] + velocity
            particles_position[i] += particles_velocity[i]
            particles_position[i] = np.clip(particles_position[i], var_min, var_max)

            cost, details = cost_function(particles_position[i])
            particles_cost[i] = cost
            details_list[i] = details
            if cost < particles_best_cost[i]:
                particles_best_cost[i] = cost
                particles_best_position[i] = particles_position[i]

        best_idx = np.argmin(particles_cost)
        gbest = {
            "position": particles_position[best_idx],
            "cost": particles_cost[best_idx],
            "details": details_list[best_idx],
        }
        print(f"Iteration {it+1}: Best Cost = {gbest['cost']}")

        if callable(callback):
            callback({
                "it": it + 1,
                "gbest": gbest,
                "pop": [
                    {"position": p.copy(), "cost": c, "details": d}
                    for p, c, d in zip(particles_position, particles_cost, details_list)
                ],
            })

    return gbest, [
        {"position": p.copy(), "cost": c, "details": d}
        for p, c, d in zip(particles_position, particles_cost, details_list)
    ]
