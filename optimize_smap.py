import math
import numpy as np
import os

from multiprocessing import Manager
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO

from problems import SMAPProblem, listener


def get_X(pop_size, n_var, config):
    """
    Create a square of size s*s = pop_size with starting positions for
    each particle.
    First axis (number of subclasses) is split into linear partitions
    Second axis (window size) is split into exponential partitions
    """
    # a = xl
    # b = (xu/xl)^(1/(pop_size - 1))
    # for i in range(pop_size):
    #   y = a * b^i
    X = np.empty((pop_size, n_var), dtype=np.float64)
    s = math.sqrt(pop_size)
    assert s % 2 == 0
    s = int(s)
    a = config["xl"][1]
    b = (config["xu"][1] / config["xl"][1]) ** (1 / (s - 1))
    s0 = (config["xu"][0] - config["xl"][0]) / s
    c = 0
    for i in range(s):
        x1 = round(a * b ** i)
        for j in range(s):
            x0 = int(config["xl"][0] + j * s0)
            X[c, 0] = float(x0)
            X[c, 1] = float(x1)
            c += 1
    return X


# Configuration of experiment
pop_size = 100
n_threads = 5
n_gens = 10
n_var = 2
_dir = "/home/marisn/Projekti/i.svm and r.smooth/"
csv_file = _dir + "train_smap.csv"
final_file = _dir + "final_smap"
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
    "xu": [32, 4096],
    "n_var": n_var,
}
# Don't let i.gensigset to fight for CPU
os.environ["OMP_THREAD_LIMIT"] = "2"

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
