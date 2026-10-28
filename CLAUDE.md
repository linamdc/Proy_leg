# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wearable Robotics course project (leg). Simulates lower-limb biomechanics with the MyoSuite `myo_sim` musculoskeletal models in MuJoCo. Code, comments and output are written in Spanish — keep that convention.

Layout that matters:
- `scripts/` — standalone analysis scripts (the real work lives here).
- `myo_sim/` — a nested clone of https://github.com/MyoHub/myo_sim (its own git repo, untracked by this one). Treat it as a third-party dependency; it has its own `CLAUDE.md`/`AGENTS.md` and `docs/wiki/` if you ever need to change it. Do not commit it into this repo.
- `notebooks/assets/myolegs*.xml` — copies of the leg model XMLs (they reference meshes via `meshdir=".."`, so they don't load standalone). `notebooks/rango_pierna.ipynb` is currently empty.
- `data/`, `docs/`, `firmware/`, `software/src`, `software/tests` — empty placeholders.

## Setup and commands

`myo_sim` must be installed as an editable package from the nested clone:

```bash
pip install -e myo_sim
```

The existing editable install on this machine points to the old location `proy_leg/myo_sim` (outside this repo), which no longer exists, so `import myo_sim` fails until it is reinstalled. Dependencies: `mujoco` (3.x), `numpy`, `matplotlib`.

```bash
python scripts/ver_pierna.py                     # open the interactive MuJoCo viewer on myolegs
python scripts/dif_legs.py                       # compare joints/muscles of myolegs vs myolegs26
python scripts/prueba_poplitea.py                # popliteal-angle test: table + plot (writes prueba_poplitea.png to CWD)
python scripts/prueba_poplitea.py --ver severo   # also show the final pose (normal|moderado|severo)
```

There are no tests or linters configured for this repo's own code.

`README.md` (in Spanish) holds the setup instructions for teammates. Whenever a change adds a new dependency, update its "Instalación" section and the list of tested versions in the same change.

## How the simulations work

- Models are loaded by registry name: `model, data = myo_sim.load("myolegs")` (80 muscles) or `"myolegs26"`. `myo_sim.load_spec(name)` returns the editable `MjSpec` if the model needs modifying before compiling.
- `scripts/prueba_poplitea.py` does *static* analysis, not dynamic simulation: it sets `qpos` directly and calls `mj_forward`, then reads passive muscle force (activation 0) × moment arm about the knee.
  - Joint convention: `knee_angle_r = 0` is full extension, positive is flexion (0–120°); `hip_flexion_r` positive is hip flexion.
  - The MyoLeg knee has coupled DOFs (tibia/patella translations) driven by `mjEQ_JOINT` polynomial equality constraints. When setting the pose by hand, those coupled `qpos` values must be computed from the polynomial coefficients in `eq_data` — `mj_forward` won't enforce them.
  - Muscle shortening is modelled by shifting `actuator_lengthrange` down by `fraction * L0` (L0 derived from `lengthrange` and `gainprm[:2]`), so passive tension starts earlier.
  - Moment arms come from the sparse `data.actuator_moment` (use `moment_rowadr` / `moment_rownnz` / `moment_colind`).
  - Tunable parameters (`T_EXAMINADOR`, `CONDICIONES`, `ACORTADOS`, `ISQUIOTIBIALES`) are module-level constants at the top of the file.
