# Online Racing Line Planning on 3D Race Tracks
This repository provides the code for the racing line planning algorithms explained in (paper link).
This includes: 

- The (offline) generation of gg-diagrams that depend on the total Velocity $V$ and the apparent vertial acceleration $\tilde{g}$ according to [X]
- The (offline) smoothing of the 3D track according to [X]
- The (offline) generation of a global racing line using Casadi 
- The (online) generation of a local racing line using Acados

## Dependencies
All scripts have only been tested on Ubuntu 22.04.2 LTS with Python 3.10.6 and the package versions listed in [requirements.txt](requirements.txt).

1. Install Acados following https://docs.acados.org/installation/#linux-mac.

2. Install the Python interface following https://docs.acados.org/python_interface/index.html.

3. Install other used Python packages:
    ```
    pip install -r requirements.txt
    ```

## Workflow

### 1. GG-Diagram Generation

### 2. Track Data

### 3. Track Smoothing

### 4. Racing Line Generation

#### Global Racing Line (offline)

#### Local Racing Line (online)

## References
