"""

Functions for analysing the properties of Pop3 stars in reduced Pop2Prime simulation catalogues.

"""

from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from yt.data_objects.particle_filters import ParticleFilter
    from yt.data_objects.data_containers import YTDataContainer

from pathlib import Path
from contextlib import contextmanager
from time import perf_counter

import yt
from yt.data_objects.particle_filters import add_particle_filter
import numpy as np
import polars as pl
from joblib import Parallel, delayed

# top-level data directory
DATA_DIR = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")


def build_pop3_ledger(
    field_map: dict[str, tuple[str, str]],
    reduced_snap_dir: Path,
    pop3_threshold: float = 1e-10,
    save_path: Path | None = None,
) -> pl.DataFrame:
    """
    Constructs a ledger of Pop3 stars across the reduced snapshots, which can be saved as a parquet file (we
    are working in polars space). This is so we can send queries to yt for spheres, profiles, rays, etc. on the 
    chunky raw snapshots (with timeseries). 
    """
    yt.set_log_level("warning")  # avoids lots of verbose output

    frame_list = Parallel(n_jobs=-1)(delayed(create_snapshot_dataframe)(
        snapshot=snapshot, pop3_threshold=pop3_threshold, field_map=field_map
        ) for snapshot in sorted(reduced_snap_dir.glob(pattern="DD*.h5")))  # joblib preserves the order

    frame_list = [frame for frame in frame_list if frame is not None]  # filter None (from empty snaps)
    ledger: pl.DataFrame = pl.concat(frame_list)  # vertical concat

    if save_path:
        ledger.write_parquet(save_path)
        print(f"Wrote parquet file at {save_path}")

    return ledger        


def create_snapshot_dataframe(
    snapshot: Path, 
    pop3_threshold: float, 
    field_map: dict[str, tuple[str, str]], 
) -> pl.DataFrame | None:
    """
    Helper to create a dataframe from a reduced snapshot catalogue such that the loop over the catalogues
    can be parallelised.
    """
    yt.set_log_level("warning")  # avoids lots of verbose output (new subprocesses spawned)

    ds = yt.load(snapshot)
    metallicities = ds.data[("pop3", "metallicity_fraction")]
    pop3_mask = np.asarray(metallicities < pop3_threshold).nonzero()[0]

    if pop3_mask.shape[0] == 0:  # skip snaps with no pop3 stars
        print(f"{snapshot.stem}: no Pop3 stars.")
        return None

    # create pure ndarrays from unyt arrays from the field map
    columns: dict[str, np.ndarray] = {
        name: ds.data[("pop3", field)][pop3_mask].to(unit).d
        for name, (field, unit) in field_map.items()
    }
    columns["positions_unitary"] = pl.Series(columns["positions_unitary"], dtype=pl.Array(pl.Float64, 3))  # treat pos vector

    # snapshot info
    redshift = ds.current_redshift
    current_time = ds.current_time.to("Myr")

    frame = pl.DataFrame(columns).with_columns(
        pl.col("particle_indices", "ptypes").cast(pl.Int64),
        pl.lit(snapshot.stem).alias("snapshot"),
        pl.lit(float(redshift)).alias("redshift"),  # need float() to remove unyt
        pl.lit(float(current_time)).alias("current_time_myr"),
    )

    print(f"{snapshot.stem}: success.")

    return frame

# NOTE: from the sim source paper we have one star forming at z ~ 23.7 and another at z ~ 18.2 (first at DD0032)
# NOTE: from Britton we know they live ~3.5 Myr 

def _pop3(pfilter: ParticleFilter, data: YTDataContainer):
    """
    Filters particles to the conditions expected for Pop3 particles.
    """
    # stars (ptype 5): filter to supernova remnants and living stars
    pop3_remnant = (data["particle_type"] == 5) & (data["particle_mass"].in_units("Msun") < 1e-10)
    pop3_star = (data["particle_type"] == 5) & (data["particle_mass"].in_units("Msun") > 1e-3)

    # dm (ptype 1): REVIEW: unsure why dm is entering the equation here
    pop3_dm = (data["particle_type"] == 1) & (data["creation_time"] > 0) & (data["particle_mass"].in_units("Msun") > 1)

    return pop3_remnant | pop3_dm | pop3_star

add_particle_filter("pop3", function=_pop3, filtered_type="all",
                    requires=["particle_type", "creation_time", "particle_mass"])

@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager around perf_counter().
    """
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"{label} completed in {elapsed:.1f}s.")

field_map = {
    "positions_unitary": ("particle_position", "unitary"),
    "ptypes": ("particle_type", "dimensionless"),
    "particle_indices": ("particle_index", "dimensionless"),
    "masses_msun": ("particle_mass", "Msun"),
    "metallicities": ("metallicity_fraction", "dimensionless"),
    "creation_times_myr": ("creation_time", "Myr"),
}

if __name__ == "__main__":

    parquet_path = Path("pop3_catalogue.parquet")

    with timer("Build ledger"):
        build_pop3_ledger(
            field_map=field_map,
            reduced_snap_dir=DATA_DIR / "pop3",
            save_path=parquet_path,
        )
    yt.set_log_level("info")
    
    # probe Pop3 stars
    test_snap = DATA_DIR / "DD0157" / "DD0157"

    ledger = pl.read_parquet(parquet_path)
    print(f"Total Pop3 Stars: {ledger["particle_indices"].n_unique()}")
    snapshot_cols = ledger.filter(pl.col("snapshot") == "DD0157")
    # grab the earliest-formed Pop3 star
    star1_row = snapshot_cols.sort("creation_times_myr").row(0, named=True)  # return as dict

    with timer("Load snapshot"):
        ds = yt.load(test_snap)
        ds.add_particle_filter("pop3")

    star_centre = ds.arr(star1_row["positions_unitary"], "unitary")
    radius = (1.0, "kpc")
    print(f"{ds.domain_center.to("unitary")}")

    with timer("Build sphere"):
        sp = ds.sphere(star_centre, radius)

    with timer("Projection plot"):
        p = yt.ProjectionPlot(
            ds,
            "x",
            ("gas", "temperature"),
            center=sp.center,
            data_source=sp,
            width=(2.0, "kpc"),
            weight_field=("gas", "density"),
        )
        p.annotate_particles((1.0, "kpc"), p_size=3.0, ptype="pop3")  # should pick up the star

        p.save("cursory_plot.png")

