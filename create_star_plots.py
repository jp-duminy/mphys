"""

Routine for generating films for a specified star.

"""


import argparse
import gc
from pathlib import Path

import polars as pl
import yt
from yt.funcs import ensure_dir

from find_pop3_stars import build_film_tasks, add_metallicity3
from utils import timer, DATA_DIR

yt.enable_parallelism()

def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="star-plot",
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
        help="Width of each panel in kpc."
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=str,
        default="x",
        help="Axis to project onto."
    )
    parser.add_argument(
        "-q",
        "--quickpeek",
        action="store_true",
    )

    return parser.parse_args()

def generate_star_images(
    outdir: Path,
    ledger_path: Path,
    width_kpc: float = 1.0,
    axis: str = "x",
    quickpeek: bool = False,
) -> None:
    """
    Generates projection plots of the star in each snapshot and saves them as .h5 files to the 
    requested output directory.
    """
    ledger = pl.read_parquet(ledger_path)
    tasks = build_film_tasks(ledger=ledger, particle_indices=[334267081], snapshot_dir=DATA_DIR)  # TODO: support multiple stars

    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity3")
    ]
    weight_field = ("gas", "density")

    for row in yt.parallel_objects(list(tasks.iter_rows(named=True)), dynamic=False):

        yt.mylog.info(f"Processing {row['snapshot_path']}")

        outpath = outdir / f"{row['base_name']}_{axis}.h5"

        if outpath.exists():
            yt.mylog.info(f"{outpath} already exists.")
            continue

        ds = yt.load(row["snapshot_path"])
        add_metallicity3(ds=ds)

        centre = ds.arr(row["centre_unitary"], "unitary")
        width = ds.quan(width_kpc, "kpc")

        region = ds.box(centre - 1.05 * width / 2,
                centre + 1.05 * width / 2)

        with timer("Generate Plot:"):
            p = yt.ProjectionPlot(
                ds, axis, fields, weight_field=weight_field,
                center=centre, width=width, data_source=region)
            data = {field[1]: p.frb[field] for field in fields}

        if quickpeek:
            if yt.is_root():
                p.save(f"{outdir / 'projection_images'}/")
        del p

        if yt.is_root():
            extra_attrs = {"centre_unitary": centre.to("unitary"), "width_kpc": width.to("kpc")}
            yt.save_as_dataset(ds, filename=str(outpath), data=data, extra_attrs=extra_attrs)

        region.clear_data()
        del region
        del ds
        val = gc.collect()
        yt.mylog.info(f"Removed {val:,.2f} objects.")

if __name__ == "__main__":

    args = parse_args()
    ensure_dir(args.outdir)
    generate_star_images(
        outdir=args.outdir,
        ledger_path=Path("pop3_catalogue.parquet"),
        width_kpc=args.width,
        axis=args.axis,
        quickpeek=args.quickpeek,
    )
