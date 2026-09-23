"""

Functions for analysing the properties of Pop3 stars in reduced Pop2Prime simulation catalogues.

"""

from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from yt.data_objects.particle_filters import ParticleFilter
    from yt.data_objects.data_containers import YTDataContainer

import pickle
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from time import perf_counter

import yt
from yt.data_objects.particle_filters import add_particle_filter
import numpy as np

# top-level data directory
DATA_DIR = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")

@dataclass(slots=True)
class StarCatalogue:
    """
    Contains the following info for Pop3 stars in one catalogue:

    - particle_indices: unique cross-snap identifier
    - masses: the masses in Msun
    - metallicities: the fraction of mass which is metal (dimensionless)
    - ages: the current age of the star (Myr)
    - positions: the position of the star in the box (unitary to boxsize) (nx3)

    And snapshot information:

    - redshift: the current redshift
    """
    particle_indices: np.ndarray
    masses: np.ndarray
    metallicities: np.ndarray
    ages: np.ndarray
    positions: np.ndarray

    redshift: float

# NOTE: from the sim source paper we have one star forming at z ~ 23.7 and another at z ~ 18.2 (first at DD0032)
# NOTE: from Britton we know they live ~3.5 Myr 
def build_pop3_star_catalogues(reduced_snap_dir: Path, metal_free_threshold: float = 1e-10) -> dict[str, StarCatalogue]:
    """
    Iterates over the reduced catalogues to locate the Pop3 stars in each. 

    Returns a dict of StarCatalogues keyed by the snapshot identifier (DD****).
    """
    catalogues: dict[str, StarCatalogue] = {}
    yt.set_log_level("warning")  # avoids lots of verbose output

    # this is embarrassingly parallel but also takes 5m
    for snapshot in sorted(reduced_snap_dir.glob(pattern="DD*.h5")):  # sorted() so snapshots are time-ordered in the dict (not the case by default)
        ds = yt.load(snapshot)
        metallicities = ds.data[("pop3", "metallicity_fraction")]
        pop3_mask = np.asarray(metallicities < metal_free_threshold).nonzero()[0]

        if pop3_mask.shape[0] == 0:  # skip snaps with no pop3 stars
            print(f"{snapshot.stem}: no Pop3 stars.")
            continue

        # basic datasets
        pop3_metallicities = metallicities[pop3_mask]
        pop3_indices = ds.data[("pop3", "particle_index")][pop3_mask]
        pop3_mass = ds.data[("pop3", "particle_mass")].to("Msun")[pop3_mask]

        # stack/derived
        creation_times = ds.data[("pop3", "creation_time")].to("Myr")
        pop3_creation_time = creation_times[pop3_mask]
        pop3_ages = (ds.current_time.to("Myr") - pop3_creation_time)
        pop3_positions = np.column_stack([ds.data[("pop3", f"particle_position_{axis}")].to("unitary")[pop3_mask] for axis in ["x", "y", "z"]])

        # snapshot info
        redshift = ds.current_redshift

        cat = StarCatalogue(
            particle_indices=pop3_indices,
            masses=pop3_mass,
            metallicities=pop3_metallicities,
            ages=pop3_ages,
            positions=pop3_positions,
            redshift=redshift,
        )

        catalogues[snapshot.stem] = cat
        print(f"{snapshot.stem}: success.")

    return catalogues

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

# probe Pop3 stars
test_snap = DATA_DIR / "DD0157" / "DD0157"

yt.set_log_level("info")

with open("pop3_catalogues.pkl", "rb") as f:
    catalogues = pickle.load(f)

cat: StarCatalogue = catalogues[test_snap.stem]

print(f"Number of Pop3 Stars: {len(cat.particle_indices)}")

with timer("Load snapshot"):
    ds = yt.load(test_snap)
    ds.add_particle_filter("pop3")

star1_pos = cat.positions[0]  # test on whichever pop3 star appears first
radius = (1.0, "kpc")
print(f"{ds.domain_center.to("unitary")}")

with timer("Build sphere"):
    sp = ds.sphere(star1_pos, radius)

with timer("Projection plot"):
    p = yt.ProjectionPlot(
        ds,
        "x",
        ("gas", "temperature"),
        center=sp.center,
        data_source=sp,
        width=(2.0, "kpc"),
        weight_field="temperature",
    )
    p.annotate_particles((1.0, "kpc"), p_size=3.0, ptype="pop3")  # should pick up the star

    p.save("cursory_plot.png")

