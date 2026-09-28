"""

Routine for generating films for a specified star.

"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import yt
from yt.funcs import ensure_dir
yt.enable_parallelism()

from find_pop3_stars import build_film_tasks, add_metallicity3
from utils import timer, DATA_DIR

def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="mphys",
        description="Routine for making a plot of stars' lifetimes across the Pop2Prime simulation.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-o", 
        "--outdir",
        type=Path,
        default=Path("."),
        help="Path to output directory."
    )
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=1.0,
        help="Width of each panel in ckpc."
    )

    return parser.parse_args()

def main() -> None:
    """
    Executes the plotting routine.
    """
    ledger = pl.read_parquet(Path("pop3_catalogue.parquet"))


if __name__ == "__main__":

    args = parse_args()

    output_dir = args.outdir

    es = yt.load(DATA_DIR / "simulation.h5")
    fns = es.data["filename"].astype(str)
    rleft = es.data["RefineRegionLeftEdge"]
    rright = es.data["RefineRegionRightEdge"]

    center = es.arr([0.5]*3, "unitary")
    width = es.quan(0.15, "unitary")

    output_dir = "projections"
    ensure_dir(output_dir)

    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity3")
    ]
    wfield = ("gas", "density")
    pfields = [
        ("io", "particle_mass")
    ]

    for fn in yt.parallel_objects(fns, dynamic=True):
        ds = yt.load(os.path.join(data_dir, fn))

        ax = "x"
        ofn = os.path.join(output_dir, f"{ds.basename}_{ax}.h5")
        if os.path.exists(ofn):
            continue
        
        add_p2p_fields(ds)

        my_c = reunit(ds, center, "unitary")
        my_w = reunit(ds, width, "unitary")
        region = ds.box(my_c - 1.05 * my_w / 2,
                        my_c + 1.05 * my_w / 2)

        p = yt.ProjectionPlot(
            ds, ax, fields, weight_field=wfield,
            center=my_c, width=my_w, data_source=region)
        data = {field[1]: p.frb[field] for field in fields}
        if yt.is_root():
            p.save("projection_images/")
        del p

        p = yt.ParticleProjectionPlot(
            ds, ax, pfields,
            center=my_c, width=my_w, data_source=region)
        data.update({field[1]: p.frb[field] for field in pfields})
        if yt.is_root():
            p.save("projection_images/")
        del p

        if yt.is_root():
            extra_attrs = {"center": my_c, "width": my_w}
            yt.save_as_dataset(ds, filename=ofn, data=data, extra_attrs=extra_attrs)

        region.clear_data()
        del region
        del ds
        val = gc.collect()
        yt.mylog.info(f"Removed {val} garbages!")