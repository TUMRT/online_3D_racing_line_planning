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
- Please cite our work if you use the provided code or parts of it ([Citing](#citing)). 

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

You can visualize the gg-diagrams and its diamond-shaped underapproximations with:
```
python gg_diagram_generation/plot_gg_diagrams.py
```

Vehicle parameters can be changed or added in [data/vehicle_params](data/vehicle_params). If you add a vehicle or change the vehicle name you have to adapt the name in the above scripts as well.

### 2. Track Data
To create a 3D track according to the representation in [X] the track must be available in one of the two data formats:

1. Global $x$ and $y$ coordinates of a reference line (e.g. centerline) which is assumed to be at $z=0$, the widths to the left and right boundary (projected on the xy-plane) and the banking angle. An example for the Las Vegas Motor Speedway is given in [data/raw_track_data/LVMS_2d_centerline_banking.csv](data/raw_track_data/LVMS_2d_centerline_banking.csv). This track representation is especially suited for oval tracks with small slopes but large banking angles.
2. Global $x$, $y$, and $z$ coordinates of track boundary pairs. An example for the Mount Panorama Circuit in Bathurst is given in [data/raw_track_data/mount_panorama_3d.csv](data/raw_track_data/mount_panorama_bounds_3d.csv)

To generate the 3D track data run the script:
```
python track_processing/gen_3d_track_data.py
```
Make sure to select the correct method for the present track format (1. or 2.). This will create the track data files in [data/3d_tracks](data/3d_tracks) with the needed coordinates, euler angles, and angular velocities.

### 3. Track Smoothing
As the 3D track data generated in [2. Track Data](track-data) can be noisy a track smoothing according to [X] is performed by running the script:
```
python track_processing/smooth_3d_track.py
```
You can visualize the final tracks and its angular information with:
```
python track_processing/plot_track.py
```

### 4. Racing Line Generation
We distinguish between the global and the local racing line. The global racing line is the closed racing line around the track. It is generated offline and implemented using Casadi. A constraint of the formulated OCP is that the start state must equal the end state.

In contrast to the global racing line, the local racing line has a limited spatial horizon and is the time-optimal trajectory for the upcoming track section. It is generated online with a moving horizon and implemented using Acados.
#### Global Racing Line (offline)
To generate the global racing line run the script: 
```
python global_racing_line/gen_global_racing_line.py
```
Note that it may take some time until the OCP converges.

You can visualize the global racing line with:
```
python global_racing_line/plot_global_racing_line.py
```

#### Local Racing Line (online)

## Citing


## References
