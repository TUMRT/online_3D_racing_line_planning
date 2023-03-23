# Online Racing Line Planning on 3D Race Tracks
This repository provides the code for the racing line planning algorithms explained in (paper link).
This includes: 

- The (offline) generation of gg-diagrams that depend on the total Velocity $V$ and the apparent vertical acceleration $\tilde{g}$ according to [X]
- The (offline) underapproximation of the true gg-diagrams with diamond shapes
- The (offline) smoothing of the 3D track according to [X]
- The (offline) generation of a global racing line using Casadi according to [X]
- The (online) generation of a local racing line using Acados

## Notes

- This repository does not intend to provide a directly applicable trajectory planner, installable Python package, or ready to use ROS node. It is rather a collection of methods and algorithms used in the paper [x].
- All paths are relative. If you move a file you have to adapt the paths accordingly.

## Dependencies
All scripts have only been tested on Ubuntu 22.04.2 LTS with Python 3.10.6 and the package versions listed in [requirements.txt](requirements.txt).

1. Install Acados following https://docs.acados.org/installation/#linux-mac.

2. Install the Python interface following https://docs.acados.org/python_interface/index.html.

3. Install other used Python packages:
    ```
    pip install -r requirements.txt
    ```

## Workflow

### 1. gg-Diagram Generation
The generation of the gg-diagrams depending on $V$ and $\tilde{g}$ follows [X] and [X]. To generate the gg-diagrams run the script:
 ```
 python gg_diagram_generation/gen_gg_diagrams.py
 ```
This will create the polar-coordinate representation of the true gg-diagram shapes $\rho(V, \tilde{g}, \alpha)$ in the folder [data/gg_diagrams](data/gg_diagrams) in the vehicle and velocity frame.

To generate the diamond-shaped underapproximations of the gg-diagrams as introduced in [X] run:
```
python gg_diagram_generation/gen_diamond_representation.py
```
The resulting lookup tables for both frames will be added to [data/gg_diagrams](data/gg_diagrams) as well.

Vehicle parameters can be changed or added in [data/vehicle_params](data/vehicle_params). If you add a vehicle or change the vehicle name you have to adapt the name in the above scripts as well.

### 2. Track Data

### 3. Track Smoothing

### 4. Racing Line Generation

#### Global Racing Line (offline)

#### Local Racing Line (online)


## References
