import numpy as np

from multiprocessing import Manager
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO

from problems import SMAPProblem, listener


def get_X(pop_size, n_var, config):
    X = np.empty((pop_size, n_var), dtype=np.float64)
    s0 = (config["xu"][0] - config["xl"][0]) / pop_size
    s1 = (config["xu"][1] - config["xl"][1]) / pop_size
    for i in range(pop_size):
        X[i, 0] = int(config["xl"][0] + i * s0)
        X[i, 1] = int(config["xl"][1] + i * s1)
    return X


# Configuration of experiment
pop_size = 20
n_threads = 5
n_gens = 10
n_var = 2
csv_file = "train_smap.csv"
final_file = "final_smap"
config = {
    "group_t": "p1",
    "subgroup_t": "RGBI",
    "group_v": "p1",
    "subgroup_v": "RGBI",
    "training": "training",
    "signature": "p1_rgbi",
    "output": "p1_rgbi_smap",
    "validation": "validation",
    "xl": [2, 2],
    "xu": [30, 4096],
    "n_var": n_var,
}

# Prepare for a run
pool = ThreadPool(n_threads)
manager = Manager()
q = manager.Queue()
watcher = pool.apply_async(listener, (q, csv_file))
runner = StarmapParallelization(pool.starmap)
problem = SMAPProblem(elementwise_runner=runner, config=config, q=q)
algorithm = PSO(
    pop_size=pop_size,
    sampling=get_X(pop_size, n_var, config),
)

# Run actual search
res = minimize(problem, algorithm, termination=("n_gen", n_gens), verbose=True, seed=42)
q.put("kill")
pool.close()
pool.join()

with open(final_file, "w") as f:
    f.write(f"X0:{res.X[0]:.6} X1:{res.X[1]:.6} F:{res.F:.6}\n")
