"""

This routine generates projection plots for density, metallicity and temperature (weighted
by density) for requested stars. It is an expensive one, taking ~6 hours for one star.

"""


import argparse
import gc
from pathlib import Path

import polars as pl
import yt

from ..sim.pop3_ledger import select_stars, star_lifetime_summary, add_metallicity3
from ..utils import timer, common_parser, DATA_DIR

yt.enable_parallelism()

def build_film_tasks(
    ledger: pl.DataFrame, 
    labels: list[str], 
    post_sn_cutoff: float,
    snapshot_dir: Path
) -> pl.DataFrame:
    """
    Creates a dataframe containing the following info for making a film:

    - particle_indices: particle IDs
    - label: the label of each star for titles
    - snapshot_path: the paths to each raw snapshot
    - centre_unitary: the positions of the stars in the raw snapshot
    - base_name: the {label}_{snapshot_stem} str.

    The columns are filtered to between when the stars were born and post_sn_cutoff after death.
    """
    tasks = (
        ledger.pipe(select_stars, labels)
        .join(star_lifetime_summary(ledger), on=["particle_indices", "label"])  # adds birth/death info
        .filter(pl.col("current_time_myr") <= pl.col("first_dead_myr").fill_null(float("inf")) + post_sn_cutoff)  # filter to where we want to visualise
        .sort("particle_indices", "snapshot")  # time order
        .with_columns(  # add global plotting info
            pl.col("positions_unitary").first().over("particle_indices").alias("centre_unitary"),  # position of star in its first snap
            pl.format("{}/{}/{}", pl.lit(str(snapshot_dir)), pl.col("snapshot"), pl.col("snapshot"))  # concatenate directories
                .alias("snapshot_path"),
            pl.format("{}_{}", pl.col("label"), pl.col("snapshot")).alias("base_name"),  # concatenate star name + snap name
        )
        .select("particle_indices", "label", "snapshot_path", "centre_unitary", "base_name")
    )

    if not all(Path(path).exists() for path in tasks["snapshot_path"]):  # quick guard
        print("Warning: not all snapshot paths exist (should not happen by construction).")

    return tasks

def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        prog="star-plot",
        description="Routine for making a plot of stars' lifetimes across the Pop2Prime simulation.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=1.5,
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
        help="Saves pngs for quick inspection."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrites existing data files."
    )
    parser.add_argument(
        "-p",
        "--postdeath",
        type=float,
        default=125,
        help="Myr past death for which plots should be made."
    )

    return parser.parse_args()

def generate_star_images(
    outdir: Path,
    tasks: pl.DataFrame,
    width_kpc: float = 1.0,
    axis: str = "x",
    quickpeek: bool = False,
    override: bool = False,
) -> None:
    """
    Generates projection plots of the star in each snapshot and saves them as .h5 files to the 
    requested output directory.
    """
    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity3")
    ]
    weight_field = ("gas", "density")

    for label in tasks["label"].unique():
        (outdir / label).mkdir(parents=True, exist_ok=True)
        (outdir / label / "projection_images").mkdir(parents=True, exist_ok=True)

    for row in yt.parallel_objects(list(tasks.iter_rows(named=True)), dynamic=False):

        yt.mylog.info(f"Processing {row['snapshot_path']}")

        star_dir = outdir / row["label"]
        outpath = star_dir / f"{row['base_name']}_{axis}.h5"

        if outpath.exists() and not override:
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
                p.save(f"{star_dir / 'projection_images'}/")
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

    ledger = pl.read_parquet(args.ledger)
    labels = args.stars or star_lifetime_summary(ledger)["label"].to_list()

    tasks: pl.DataFrame = build_film_tasks(
        ledger=ledger, 
        labels=labels,
        post_sn_cutoff=args.postdeath,
        snapshot_dir=DATA_DIR
    )

    generate_star_images(
        outdir=args.outdir,
        tasks=tasks,
        width_kpc=args.width,
        axis=args.axis,
        quickpeek=args.quickpeek,
        override=args.force,
    )
