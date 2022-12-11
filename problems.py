import json

from pymoo.core.problem import ElementwiseProblem

import grass.script as grass
from grass.exceptions import CalledModuleError
from grass.script.core import tempname


class SMAPProblem(ElementwiseProblem):
    def __init__(self, config, q, **kwargs):
        super().__init__(
            n_var=config["n_var"], n_obj=1, xl=config["xl"], xu=config["xu"], **kwargs
        )
        self.config = config
        self.q = q

    def _evaluate(self, x, out, *args, **kwargs):
        postfix = tempname(10)
        nsigs = int(x[0])
        x1 = int(x[1])
        blocksize = x1 if x1 % 2 == 0 else x1 + 1
        try:
            grass.run_command(
                "i.gensigset",
                trainingmap=self.config["training"],
                group=self.config["group_t"],
                subgroup=self.config["subgroup_t"],
                signaturefile=self.config["signature"] + postfix,
                maxsig=nsigs,
                overwrite=True,
                quiet=True,
            )
            grass.run_command(
                "i.smap",
                group=self.config["group_v"],
                subgroup=self.config["subgroup_v"],
                signaturefile=self.config["signature"] + postfix,
                output=self.config["output"] + postfix,
                blocksize=blocksize,
                overwrite=True,
                quiet=True,
            )
            evaluation = grass.read_command(
                "r.kappa",
                reference=self.config["validation"],
                classification=self.config["output"] + postfix,
                format="json",
                quiet=True,
            )
        except CalledModuleError:
            print(f"\n===== FAIL WITH nsigs:{nsigs} blocksize:{blocksize}")
            self.q.put(f"{nsigs},{blocksize},NA,NA,NA")
            return
        try:
            kappa = json.loads(evaluation)
            mcc = float(kappa["mcc"])
        except (json.decoder.JSONDecodeError, KeyError, ValueError):
            self.q.put(f"{nsigs},{blocksize},NA,NA,NA")
            out["F"] = 10
            return
        if mcc < -1:
            out["F"] = 10
        else:
            out["F"] = 1 - mcc
        self.q.put(
            f"{nsigs},{blocksize},{kappa['mcc']},{kappa['kappa']},{kappa['overall_accuracy']}"
        )
        try:
            grass.read_command(
                "g.remove",
                flags="f",
                type="raster",
                name=self.config["output"] + postfix,
                quiet=True,
            )
        except CalledModuleError:
            print("err removing raster")


class SVMProblem(ElementwiseProblem):
    def __init__(self, config, q, **kwargs):
        super().__init__(
            n_var=config["n_var"], n_obj=1, xl=config["xl"], xu=config["xu"], **kwargs
        )
        self.config = config
        self.q = q

    def _evaluate(self, x, out, *args, **kwargs):
        postfix = tempname(10)
        try:
            grass.run_command(
                "i.svm.train",
                group=self.config["group_t"],
                subgroup=self.config["subgroup_t"],
                input=self.config["training"],
                signaturefile=self.config["signature"] + postfix,
                cost=x[0],
                gamma=x[1],
                overwrite=True,
                quiet=True,
            )
            grass.run_command(
                "i.svm.predict",
                group=self.config["group_v"],
                subgroup=self.config["subgroup_v"],
                signaturefile=self.config["signature"] + postfix,
                output=self.config["output"] + postfix,
                overwrite=True,
                quiet=True,
            )
            evaluation = grass.read_command(
                "r.kappa",
                reference=self.config["validation"],
                classification=self.config["output"] + postfix,
                format="json",
                quiet=True,
            )
        except CalledModuleError:
            print(f"\n===== FAILURE WITH C:{x[0]} gamma:{x[1]}\n")
            out["F"] = 10
            self.q.put(f"{x[0]},{x[1]},NA,NA,NA")
            return
        try:
            kappa = json.loads(evaluation)
            mcc = float(kappa["mcc"])
        except (json.decoder.JSONDecodeError, KeyError, ValueError):
            self.q.put(f"{x[0]},{x[1]},NA,NA,NA")
            out["F"] = 10
            return
        if mcc < -1:
            out["F"] = 10
        else:
            out["F"] = 1 - mcc
        self.q.put(
            f"{x[0]},{x[1]},{kappa['mcc']},{kappa['kappa']},{kappa['overall_accuracy']}"
        )
        try:
            grass.read_command(
                "g.remove",
                flags="f",
                type="raster",
                name=self.config["output"] + postfix,
                quiet=True,
            )
        except CalledModuleError:
            print("err removing raster")


def listener(q, fn):
    with open(fn, "w", buffering=1) as f:
        f.write("X0,X1,mcc,kappa,oac\n")
        while 1:
            m = q.get()
            if m == "kill":
                break
            f.write(str(m) + "\n")
            f.flush()
