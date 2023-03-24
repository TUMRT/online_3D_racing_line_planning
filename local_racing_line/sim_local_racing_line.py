import os
import sys
import casadi as ca
import numpy as np
import yaml
import pandas as pd

params = {
    'track_name': 'LVMS_3d_smoothed.csv',
    # 'track_name': 'mount_panorama_3d_smoothed.csv',
    'vehicle_name': 'dallaraAV21',
    'gg_mode': 'diamond',  # polar or diamond
    'gg_margin': 0.0,
    'safety_distance': 0.5,  # in meter
    'optimization_horizon': 500.0,  # in meter
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + params['vehicle_name'] + '.yml')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams', params['vehicle_name'], 'velocity_frame')
track_path = os.path.join(data_path, 'smoothed_track_data')
raceline_out_path = os.path.join(data_path, 'global_racing_lines')
sys.path.append(os.path.join(dir_path, '..', 'src'))

# load vehicle and tire parameters
with open(vehicle_params_path, 'r') as stream:
    params.update(yaml.safe_load(stream))

from track3D import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model

track_handler = Track3D(
        path=os.path.join(track_path, params['track_name'])
    )

gg_handler = GGManager(
    gg_path=gg_diagram_path,
    gg_margin=params['gg_margin']
)

point_mass_model = export_point_mass_ode_model(
        vehicle_params=params['vehicle_params'],
        track_handler=track_handler,
        gg_handler=gg_handler,
        optimization_horizon=params['optimization_horizon'],
        gg_mode=params['gg_mode']
)

local_raceline_planner = LocalRacinglinePlanner(
    params=params,
    track_handler=track_handler,
    gg_handler=gg_handler,
    model=point_mass_model,
    optimization_horizon=params['optimization_horizon'],
    gg_mode=params['gg_mode'],
)