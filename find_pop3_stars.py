"""

Functions for analysing the properties of Pop3 stars in reduced Pop2Prime simulation catalogues.

"""

import pickle
from pathlib import Path
from dataclasses import dataclass

import yt
import numpy as np

# top-level data directory
data_dir = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")

# exploratory dataset at z = 17.7
ds = yt.load(data_dir / "pop3" / "DD0157.h5")
print(ds.field_list)
print(ds.data[("pop3", "metallicity_fraction")])
print(ds.data[("pop3", "particle_position_x")])
print(ds.data[("pop3", "creation_time")].to("Gyr"))  # this and above show our two stars in the snapshot

@dataclass(slots=True)
class StarCatalogue:
    """
    Contains the following info for Pop3 stars in one catalogue:

    - particle_index: unique cross-snap identifier
    - metallicity: the fraction of mass which is metal (dimensionless)
    - age: the current age of the star (Myr)
    - position: the position of the star in the box (physical kpc h^-1) (nx3)
    """
    particle_indices: np.ndarray
    masses: np.ndarray
    metallicities: np.ndarray
    ages: np.ndarray
    positions: np.ndarray

# NOTE: from the sim source paper we have one star forming at z ~ 23.7 and another at z ~ 18.2 
# NOTE: from Britton we know they live ~3.5 Myr 
def build_pop3_star_catalogues(reduced_snap_dir: Path, metal_free_threshold: float = 1e-10) -> dict[str, StarCatalogue]:
    """
    Iterates over the reduced catalogues to locate the Pop3 stars in each. 

    Returns a dict of StarCatalogues keyed by the snapshot identifier (DD****).
    """
    catalogues: dict[str, StarCatalogue] = {}
    yt.set_log_level("warning")  # avoids lots of verbose output

    for snapshot in reduced_snap_dir.glob(pattern="DD*.h5"):
        ds = yt.load(snapshot)
        metallicities = ds.data[("pop3", "metallicity_fraction")]
        pop3_mask = np.asarray(metallicities < metal_free_threshold).nonzero()[0]

        if pop3_mask.shape[0] == 0:  # skip snaps with no pop3 stars
            continue

        # basic datasets
        pop3_metallicities = ds.data[("pop3", "metallicity_fraction")][pop3_mask]
        pop3_indices = ds.data[("pop3", "particle_index")][pop3_mask]
        pop3_mass = ds.data[("pop3", "particle_mass")].to("Msun")[pop3_mask]

        # stack/derived
        creation_times = ds.data[("pop3", "creation_time")].to("Myr")
        pop3_creation_time = creation_times[pop3_mask]
        pop3_ages = (ds.current_time.to("Myr") - pop3_creation_time)
        pop3_positions = np.column_stack([ds.data[("pop3", f"particle_position_{axis}")].to("kpc")[pop3_mask] for axis in ["x", "y", "z"]])

        cat = StarCatalogue(
            particle_indices=pop3_indices,
            masses=pop3_mass,
            metallicities=pop3_metallicities,
            ages=pop3_ages,
            positions=pop3_positions,
        )

        catalogues[snapshot.stem] = cat
        print(f"{snapshot.stem}: success.")

    return catalogues

catalogues = build_pop3_star_catalogues(reduced_snap_dir=data_dir / "pop3")
        
with open("pop3_catalogues.pkl", "wb") as f:  # needs wb for write-binary
    pickle.dump(catalogues, f)
