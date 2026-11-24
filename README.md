# NSInduct

NSInduct studies magnetic winding, Maxwell-stress feedback and dissipation in an axisymmetric spherical shell. A prescribed, confined poloidal field winds under differential rotation. The evolving toroidal field exerts a Maxwell torque on the flow. The model assumes constant density and includes toroidal magnetic diffusion and optional viscosity.

A conservative finite-difference solver uses classical RK4. An independent Dedalus v3 solver uses a spherical shell basis, toroidal vectors and RK443. The [model and benchmarks](SCIENCE.md) give the equations, boundary conditions and comparisons.

At magnetic diffusivity `2e-5`, the first sampled time at which differential-rotation kinetic energy falls to half its initial value is:

| RMS poloidal field | Finite differences (48 × 72) | Dedalus (24 × 36) |
| ---: | ---: | ---: |
| 0.035 | 16.5 | 16.4 |
| 0.070 | 8.3 | 8.3 |
| 0.140 | 4.2 | 4.2 |

Time is measured in inverse initial-shear units and sampled every 0.1. The half-times scale approximately as the inverse seed field over this range. They characterize energy transfer in this shell. The model excludes stellar structure, meridional circulation, evolving poloidal flux and nonaxisymmetric dynamics. Late ideal wall traces and polar extrema remain resolution-sensitive.

## Installation and reproduction

The core package requires Python 3.11 or later. For finite differences, run from the repository root:

```bash
python -m venv ../nsinduct-env
source ../nsinduct-env/bin/activate
python -m pip install '.[test]'
python -m pytest -m 'not dedalus'
python scripts/reproduce_results.py --output-dir ../nsinduct-output/fd
```

The equivalent core Conda specification is [environment.yml](environment.yml). For Dedalus and the notebooks, use Conda with conda-forge. The [Dedalus specification](environment-dedalus.yml) selects Python 3.12, Dedalus 3.0.5, OpenMPI, FFTW and HDF5 through Conda. Run the solver in one MPI process.

```bash
conda env create -f environment-dedalus.yml
conda activate nsinduct-dedalus
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
python -m pip install --no-deps --no-build-isolation .
python -m pytest -ra
python scripts/reproduce_results.py --notebooks-only --output-dir ../nsinduct-output/notebooks
python scripts/reproduce_dedalus.py --suite quick --output-dir ../nsinduct-output/spectral
python scripts/verify_spectral.py --matched --braking --output-dir ../nsinduct-output/verification
python scripts/verify_spectral.py --reference-dir ../nsinduct-output/verification/matched_reference --output-dir ../nsinduct-output/figures
```

The [notebooks](notebooks/) cover [kinematic winding](notebooks/01_kinematic_winding.ipynb), [magnetic braking](notebooks/02_magnetic_braking.ipynb) and [resistive decay and convergence](notebooks/03_resistivity_and_convergence.ipynb). The solvers and reconstruction code are in `src/nsinduct/`, numerical checks in `tests/`, and reproduction commands in `scripts/`. All generated data go to the specified external directories.

Optional PyVista/VTK figures and GIFs reconstruct the axisymmetric snapshots in three dimensions. The renderer needs an off-screen graphics context. Add `--backend dedalus` in an environment with both Dedalus and the rendering dependencies.

```bash
python -m pip install '.[viz]'
python scripts/render_3d.py --scene all --output-dir ../nsinduct-output/visualizations
```

## Citation

Cite Daniil Lysenko and the version used, as listed in [CITATION.cff](CITATION.cff). The source carries the [BSD 3-Clause license](LICENSE).