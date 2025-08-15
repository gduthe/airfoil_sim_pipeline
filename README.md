# Airfoil Simulation Pipeline

A comprehensive CFD pipeline for automated airfoil simulation using OpenFOAM. This pipeline handles airfoil generation, mesh creation, boundary condition setup, and CFD execution for aerodynamic analysis.

## Project Structure

```
airfoil_sim_pipeline/
├── airfoil_generation/          # Airfoil shape generation and sampling
│   ├── sample_catalog_airfoils.py
│   └── clean_airfoil_database_v3.tar
├── airfoil_meshing/             # Mesh generation for airfoils
│   ├── mesh_airfoil.py
│   ├── mesher.py
│   └── auxiliary_postpycess.py
├── boundary_condition_generation/ # BC generation and configuration
│   ├── airfoil_bc_generator.py
│   ├── turbine_bc_generator.py
│   ├── generate_bcs.py
│   └── bc_config.yml
├── cfd_running/                 # CFD execution and OpenFOAM integration
│   ├── cfd_runner.py
│   ├── run_cfd_job.py
│   ├── base.py
│   ├── cfd_config.yml
│   └── etc/                     # OpenFOAM templates and utilities
│       ├── 2D_foil_template/
│       ├── calculateCp2D
│       └── uniform_field_template/
├── main.py                      # Main pipeline execution script
├── run_one_shape.py            # Single airfoil simulation script
├── config_airf.yml             # Configuration for airfoil pipeline
├── config_turb.yml             # Configuration for turbine conditions
└── ros_config.yml              # Configuration for single shape runs
```

## Features

- **Airfoil Generation**: Sample from catalog database or generate random shapes
- **Automated Meshing**: Support for unstructured, hyperbolic, elliptical, and snappy meshes
- **Boundary Condition Generation**: Parametric BC generation with turbine operating conditions
- **CFD Execution**: OpenFOAM-based simulations with various RANS models
- **Parallel Processing**: Multi-threaded execution for batch simulations
- **Parameter Sweeps**: Support for sweeping flow conditions (AoA, Re, Ma, etc.)

## Dependencies

- Python 3.x
- OpenFOAM
- Required Python packages:
  - `numpy`
  - `pandas` 
  - `joblib`
  - `click`
  - `pyyaml`
  - `python-box`
  - `matplotlib` (for plotting)

## Quick Start

### Running a Single Airfoil Simulation

For simulating a single airfoil shape:

```bash
python run_one_shape.py -c ros_config.yml -f runs/single_test
```

### Running the Full Pipeline

For batch simulation of multiple airfoils:

```bash
python main.py -na 10 -ns 5 -c config_airf.yml -f runs/batch_test -nt 4
```

## Configuration

### Main Pipeline Configuration (`config_airf.yml`)

- **Airfoil Settings**: Catalog path, generation type, angle of attack ranges
- **Mesh Settings**: Mesh type and parameters
- **Inflow Settings**: Velocity, turbulence, Reynolds number ranges
- **CFD Settings**: Solver, turbulence model, convergence criteria

### Single Shape Configuration (`ros_config.yml`)

- **Airfoil Settings**: Path to specific .dat file
- **Mesh Settings**: Detailed mesh parameters (h_a, h_0, R_b, etc.)
- **Boundary Conditions**: Specific flow conditions
- **CFD Settings**: Solver configuration

## Command Line Options

### Main Pipeline (`main.py`)

- `-na, --num_airfoils`: Number of airfoils to simulate (default: 10)
- `-ns, --num_sims`: Number of simulations per airfoil (default: 1)
- `-c, --config_file`: Path to YAML config file (default: config.yml)
- `-f, --run_folder`: Output directory for results
- `-nt, --num_threads`: Number of parallel threads (default: 4)
- `-sbc, --same_bcs`: Use same boundary conditions for all airfoils
- `-ot, --on_turbine`: Use turbine operating conditions
- `-sm, --sweep_mode`: Parameter to sweep (u, ti, aoa, Re, Ma)
- `-sb, --sweep_bounds`: Bounds for parameter sweep
- `-p, --plot`: Show plots during execution
- `-o, --overwrite`: Overwrite existing run folder

### Single Shape (`run_one_shape.py`)

- `-c, --config_file`: Path to YAML config file (default: ros_config.yml)
- `-f, --run_folder`: Output directory (default: runs/test_run)
- `-p, --plot`: Show plots during execution
- `-o, --overwrite`: Overwrite existing run folder

## Output Structure

Simulation results are organized as follows:

```
output_folder/
├── meshes/                 # Generated meshes
│   ├── mesh_0/
│   ├── mesh_1/
│   └── ...
├── simulations/           # CFD results
│   ├── simulation_000/
│   ├── simulation_001/
│   └── ...
└── etc/                   # Metadata and configuration
    ├── airfoil_names.csv
    ├── run_info.csv
    └── run_config.yml
```

## Citation

If you use this pipeline in your research, please cite one:

```bibtex
@article{duthe2023graph,
  title={Graph neural networks for aerodynamic flow reconstruction from sparse sensing},
  author={Duth{\'e}, Gregory and Abdallah, Imad and Barber, Sarah and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2301.03228},
  year={2023}
}
```
or 

```bibtex
@article{duthe2025graph,
  title={Graph Transformers for inverse physics: reconstructing flows around arbitrary 2D airfoils},
  author={Duth{\'e}, Gregory and Abdallah, Imad and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2501.17081},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
