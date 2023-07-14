import time

import grass.script as grass
from grass.exceptions import CalledModuleError

# These two files will be written in the current directory unless full path is given
result_file = "timing_reg_all.csv"
sig_ml2 = "sig_ml2_best"

# GRASS names – no path needed
sig_mlc = "p1_rgbi_default"
sig_smap = "p1_rgbi_best"
sig_svm = "p1_rgbi_best"
tmp_raster = "rm_timing_raster"
group = "p2"
subgroup = "RGBI"
# A raster map to align with
s2_raster = "S2B_20190612.red"

# Prepare static data
grass.run_command(
    "g.region",
    region="p2",
    quiet=True,
)
grass.run_command(
    "g.remove",
    type="raster",
    name="MASK",
    flags=("f",),
    quiet=True,
)
grass.run_command(
    "v.to.rast",
    input="p2_lauki",
    output="p2_lauki",
    use="attr",
    attribute_column="pcn",
    label_column="PRODUCT_DE",
    quiet=True,
    overwrite=True,
)

with open(result_file, "w+", buffering=1) as out:
    out.write("n,module,tp,rt,cells\n")
    for i in range(5):
        # Repeat x times to provide multiple runs with each size for statistics

        # Start from the centre to test performance of prediction part
        # with an ever increasing cell count
        grass.run_command(
            "g.region",
            align=s2_raster,
            n=6261000,
            s=6260460,
            e=681000,
            w=680460,
            quiet=True,
        )
        n = 1
        while n < 23:
            # Outside of try as will return an error if MASK is missing
            grass.run_command(
                "g.remove",
                type="raster",
                name="MASK",
                flags=("f",),
                quiet=True,
            )
            try:
                univar = grass.parse_command("r.univar", map=tmp_raster, flags="g")
                cells = univar["n"]
                # Prepare input raster with n points
                # Point locations with each n will differ as the
                # computational region has changed
                grass.run_command(
                    "r.random",
                    input="p2_lauki",
                    npoints=n,
                    raster=tmp_raster,
                    seed=42,
                    overwrite=True,
                    quiet=True,
                )
                # Use sparse raster for both training and prediction
                # Yes, this is a huge IO penality for small n
                grass.run_command(
                    "g.copy",
                    rast=(
                        tmp_raster,
                        "MASK",
                    ),
                    quiet=True,
                )
                # Time MLC training
                st = time.time()
                grass.run_command(
                    "i.gensig",
                    trainingmap=tmp_raster,
                    group=group,
                    subgroup=subgroup,
                    signaturefile="remove_timing",
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},MLC,t,{et-st},{cells}\n")
                # Time SMAP training
                st = time.time()
                grass.run_command(
                    "i.gensigset",
                    trainingmap=tmp_raster,
                    group=group,
                    subgroup=subgroup,
                    signaturefile="remove_timing",
                    maxsig=30,
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},SMAP,t,{et-st},{cells}\n")
                # Time SVM training
                # At first without shrinking
                st = time.time()
                grass.run_command(
                    "i.svm.train",
                    group=group,
                    subgroup=subgroup,
                    trainingmap=tmp_raster,
                    signaturefile="remove_timing",
                    cost=29453,
                    gamma=26,
                    flags=("s",),
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},SVMn,t,{et-st},{cells}\n")
                # Second run is with shrinking
                st = time.time()
                grass.run_command(
                    "i.svm.train",
                    group=group,
                    subgroup=subgroup,
                    trainingmap=tmp_raster,
                    signaturefile="remove_timing",
                    cost=29453,
                    gamma=26,
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},SVMh,t,{et-st},{cells}\n")
                # Time ML2 training
                st = time.time()
                grass.run_command(
                    "r.learn.train",
                    group=group,
                    training_map=tmp_raster,
                    save_model=sig_ml2,
                    model_name="SVC",
                    c=29453,
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},ML2,t,{et-st},{cells}\n")
            except CalledModuleError as e:
                print(e)
                break
            try:
                # Time MLC prediction
                st = time.time()
                grass.run_command(
                    "i.maxlik",
                    group=group,
                    subgroup=subgroup,
                    signaturefile=sig_mlc,
                    output="remove_timing",
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},MLC,p,{et-st},{cells}\n")
                # Time SMAP prediction
                st = time.time()
                grass.run_command(
                    "i.smap",
                    group=group,
                    subgroup=subgroup,
                    signaturefile=sig_smap,
                    blocksize=194,
                    output="remove_timing",
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},SMAP,p,{et-st},{cells}\n")
                # Time SVM prediction
                st = time.time()
                grass.run_command(
                    "i.svm.predict",
                    group=group,
                    subgroup=subgroup,
                    signaturefile=sig_svm,
                    output="remove_timing",
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},SVM,p,{et-st},{cells}\n")
                st = time.time()
                grass.run_command(
                    "r.learn.predict",
                    group=group,
                    output="remove_timing",
                    load_model=sig_ml2,
                    overwrite=True,
                    quiet=True,
                )
                et = time.time()
                out.write(f"{n},ML2,p,{et-st},{cells}\n")
            except CalledModuleError as e:
                print(e)
                break
            # Grow computation region to increase cell count for predictor
            grass.run_command(
                "g.region",
                grow=50,
                align=s2_raster,
                quiet=True,
            )
            # One run complete
            n += 1
