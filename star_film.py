"""

Routine for generating films for a specified star.

"""

import argparse
from pathlib import Path
import gc

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

def main() -> None:
    """
    Executes the plotting routine.
    """
    args = parse_args()
    output_dir: Path = args.outdir
    ensure_dir(output_dir)
    ledger = pl.read_parquet(Path("pop3_catalogue.parquet"))
    
    tasks = build_film_tasks(ledger=ledger, particle_indices=[334267081], snapshot_dir=DATA_DIR)

    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity3")
    ]
    weight_field = ("gas", "density")

    for row in yt.parallel_objects(list(tasks.iter_rows(named=True)), dynamic=False):

        yt.mylog.info(f"Processing {row['snapshot_path']}")

        outpath = output_dir / f"{row['base_name']}_{args.axis}.h5"

        if outpath.exists():
            yt.mylog.info(f"{outpath} already exists.")
            continue

        ds = yt.load(row["snapshot_path"])
        add_metallicity3(ds=ds)

        centre = ds.arr(row["centre_unitary"], "unitary")
        width = ds.quan(args.width, "kpc")

        region = ds.box(centre - 1.05 * width / 2,
                centre + 1.05 * width / 2)

        with timer("Generate Plot:"):
            p = yt.ProjectionPlot(
                ds, args.axis, fields, weight_field=weight_field,
                center=centre, width=width, data_source=region)
            data = {field[1]: p.frb[field] for field in fields}

        if args.quickpeek:
            if yt.is_root():
                p.save(f"{output_dir / 'projection_images'}/")
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

    main()
