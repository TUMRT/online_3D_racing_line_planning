import os
import sys
import yaml
import numpy as np
import casadi as ca


params = {
    'track_name': 'LVMS_3d_smoothed.csv',
    'optimization_horizon': 500.0,  # in meter
    # 'track_name': 'mount_panorama_3d_smoothed.csv',
    # 'optimization_horizon': 300.0,  # in meter
    'vehicle_name': 'dallaraAV21',
    'gg_mode': 'polar',  # polar or diamond
    'gg_margin': 0.0,
    'safety_distance': 0.5,  # in meter
    'V_max': None,  # V_max in m/s (set to None for limitless)
    'V_max_after': 40.0  # apply V_max after x seconds (simulation time)
}
# next state assuming perfect tracking of the racing line and 100ms calculation time
assumed_calc_time = 0.1  # needed since the visualization is not the fastest

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
from visualizer import Visualizer


# create instances
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

visualizer = Visualizer(
    track_handler=track_handler,
    gg_handler=gg_handler,
    vehicle_params=params['vehicle_params'],
    threeD=False,
)

# simulation start state
s = 0.0
n = 0.0
V = 0.0
chi = 0.0
ax = 0.0
ay = 0.0
racing_line = None
sim_time = 0.0

while 1:
    V_max = 1e3
    if params['V_max'] is not None and sim_time > params['V_max_after']:
        V_max = params['V_max']

    racing_line = local_raceline_planner.calc_raceline(
        s=s,
        V=V,
        n=n,
        chi=chi,
        ax=ax,
        ay=ay,
        safety_distance=params['safety_distance'],
        prev_solution=racing_line,
        V_max=V_max
    )

    visualizer.update(
        s=s,
        V=V,
        n=n,
        chi=chi,
        ax=ax,
        ay=ay,
        racing_line=racing_line
    )

    # ensure monotonically increasing s vector
    s_array_unwrap = np.unwrap(
        racing_line['s'],
        discont=track_handler.s[-1] / 2.0,
        period=track_handler.s[-1]
    )
    s = np.interp(assumed_calc_time, racing_line['t'], s_array_unwrap) % track_handler.s[-1]
    n = np.interp(assumed_calc_time, racing_line['t'], racing_line['n'])
    V = np.interp(assumed_calc_time, racing_line['t'], racing_line['V'])
    chi = np.interp(assumed_calc_time, racing_line['t'], racing_line['chi'])
    ax = np.interp(assumed_calc_time, racing_line['t'], racing_line['ax'])
    ay = np.interp(assumed_calc_time, racing_line['t'], racing_line['ay'])

    sim_time += assumed_calc_time

# EOF
