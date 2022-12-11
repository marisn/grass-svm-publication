import numpy as np

from multiprocessing import Manager
from multiprocessing.pool import ThreadPool

from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO

from problems import SVMProblem, listener


def get_X(pop_size, n_var, config):
    # TODO: make universal (see SMAP get_X)
    X = np.empty((pop_size, n_var), dtype=np.float64)
    s = int(math.sqrt(pop_size))
    C = 0.1
    Cs = []
    Gs = []
    for i in range(s):
        G = 0.5
        for j in range(s):
            Cs.append(C)
            Gs.append(G)
            G = G * 4
        C = C * 10
    for i in range(pop_size):
        X[i, 0] = Cs[i]
        X[i, 1] = Gs[i]
    return X


# Configuration of experiment
pop_size = 20
n_threads = 5
n_gens = 10
n_var = 2
_dir = ""
csv_file = _dir + "train_svm.csv"
final_file = _dir + "final_svm"
config = {
    "group_t": "p1",
    "subgroup_t": "RGBI",
    "group_v": "p1",
    "subgroup_v": "RGBI",
    "training": "training",
    "signature": "p1_rgbi",
    "output": "p1_rgbi_svm_c_rbf",
    "validation": "validation",
    "xl": [0.1, 0.1],
    "xu": [10000000.0, 100000.0],
    "n_var": n_var,
}

# Prepare for a run
pool = ThreadPool(n_threads)
manager = Manager()
q = manager.Queue()
watcher = pool.apply_async(listener, (q, csv_file))
runner = StarmapParallelization(pool.starmap)
problem = SVMProblem(elementwise_runner=runner, config=config, q=q)
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
