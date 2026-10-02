"""

Scripts for rudimentary testing of cloudy scripts.

"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

mphys_dir = Path.home() / "mphys"
plots_dir = mphys_dir / "plots"
mnras_style = Path.home() / "mnras.mplstyle"

plt.style.use(mnras_style)

datasets = {"default": mphys_dir / "hii_coolstar_default.ovr",
            "pop3-proxy": mphys_dir / "hii_coolstar_mod.ovr",
        }

fig, ax = plt.subplots()

for name, filepath in datasets.items():

    depth, temp = np.loadtxt(filepath, skiprows=1, usecols=(0, 1), unpack=True)
    ax.plot(depth, temp, label=f"{name} settings")

ax.set_ylabel(r"Temperature [$K$]")
ax.set_xlabel(r"Depth [$cm$]")
ax.set_title(r"Cloudy hii\_coolstar Example With default \& pop3-proxy Settings")
ax.legend()

fig.savefig(plots_dir / "example.png", dpi=400)
