import os
import sys
import yaml
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import casadi as ca
mpl.rcParams['lines.linewidth'] = 2


params = {
    #'track_name': 'LVMS_3d_smoothed.csv',
    #'optimization_horizon': 500.0,  # in meter
    'track_name': 'mount_panorama_3d_smoothed.csv',
    'optimization_horizon': 300.0,  # in meter
    'vehicle_name': 'dallaraAV21',
    'gg_mode': 'diamond',  # polar or diamond
    'gg_margin': 0.0,
    'safety_distance': 0.5,  # in meter
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

plt.ion()
fig = plt.figure()
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], figure=fig)

axis_track = plt.subplot(gs[:, 0])
axis_track.set_xlabel("x")
axis_track.set_ylabel("y")
axis_track.set_aspect('equal')
axis_track.grid()
# plot track bounds
left, right = track_handler.get_track_bounds()
axis_track.plot(left[0], left[1], color='k')
axis_track.plot(right[0], right[1], color='k')
rl_line, = axis_track.plot([0], [0], color='r')

axis_traj = plt.subplot(gs[0, 1])
axis_traj.set_xlabel("t")
axis_traj.grid()
axis_traj.set_xlim([0.0, 1.0])
axis_traj.set_ylim([0.0, gg_handler.V_max])
V_line, = axis_traj.plot([0], [0], color='r', label='$V$')
ax_line, = axis_traj.plot([0], [0], color='b', label='$\hat{a}_\mathrm{x}$')
ay_line, = axis_traj.plot([0], [0], color='g', label='$\hat{a}_\mathrm{y}$')
axis_traj.legend()

axis_gg = plt.subplot(gs[1, 1])
axis_gg.set_title('gg-Diagram')
axis_gg.set_xlabel(r"$\tilde{a}_\mathrm{y}$")
axis_gg.set_ylabel(r"$\tilde{a}_\mathrm{x}$")
diamond_line1, = axis_gg.plot([0], [0], color='g', label='Diamond-shaped underapproximation')
diamond_line2, = axis_gg.plot([0], [0], color='g')
rho_line, = axis_gg.plot([0], [0], color='b', alpha=0.5, label='Polar coordinates (exact)', linestyle='--')
axy_marker, = axis_gg.plot([0], [0], color='r', marker='o', markersize=10)
axis_gg.grid()
axis_gg.set_xlim([-40, 40])
axis_gg.set_ylim([-40, 20])
axis_gg.legend(loc='lower right')
V_text = axis_gg.text(-35.0, 15.0, "")
gt_text = axis_gg.text(-35.0, 10.0, "")

# simulation
s = 0.0
n = 0.0
V = 0.0
chi = 0.0
ax = 0.0
ay = 0.0
racing_line = None

while 1:
    ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
        V=max(V, 1.0),
        n=n,
        chi=chi,
        ax=ax,
        ay=ay,
        s=s,
        h=params['vehicle_params']['h'],
    )
    gt_text.set_text(fr'$\tilde{{g}}={float(g_tilde):.2f}$')
    V_text.set_text(fr'$V={V:.2f}$')

    # plot diamond shape for current driving state
    acc_max = gg_handler.acc_interpolator(ca.vertcat(V, g_tilde))
    gg_exponent = float(acc_max[0])
    ax_min = float(acc_max[1])
    ax_max = float(acc_max[2])
    ay_max = float(acc_max[3])
    ay_array = np.linspace(-ay_max, ay_max, 100)
    ax_array = - ax_min * np.power(
        1.0 - np.power(np.abs(ay_array) / ay_max, gg_exponent),
        1.0 / gg_exponent,
    )
    diamond_line1.set_xdata([ay_array])
    diamond_line1.set_ydata([np.minimum(ax_array, ax_max)])
    diamond_line2.set_xdata([ay_array])
    diamond_line2.set_ydata([-ax_array])

    # plot exact shape of gg-diagram for current driving state
    alpha_array = np.linspace(-np.pi, np.pi, 100)
    rho_array = gg_handler.rho_interpolator(np.array([V * np.ones_like(alpha_array), float(g_tilde) * np.ones_like(alpha_array), alpha_array])).full().squeeze()
    rho_line.set_xdata([np.cos(alpha_array) * rho_array])
    rho_line.set_ydata([np.sin(alpha_array) * rho_array])

    axy_marker.set_xdata([float(ay_tilde)])
    axy_marker.set_ydata([float(ax_tilde)])

    racing_line = local_raceline_planner.calc_raceline(
        s=s,
        V=V,
        n=n,
        chi=chi,
        ax=ax,
        ay=ay,
        safety_distance=params['safety_distance'],
        prev_solution=racing_line,
    )


    axis_traj.set_xlim([0.0, racing_line['t'][-1]*1.05])
    axis_traj.set_ylim([
        min(0.0, 1.05*np.min(np.concatenate((racing_line['ax'], racing_line['ay'])))),
        1.05*np.max(np.concatenate((racing_line['V'], racing_line['ax'], racing_line['ay'])))
    ])
    rl_line.set_xdata(racing_line['x'])
    rl_line.set_ydata(racing_line['y'])
    V_line.set_xdata(racing_line['t'])
    V_line.set_ydata(racing_line['V'])
    ax_line.set_xdata(racing_line['t'])
    ax_line.set_ydata(racing_line['ax'])
    ay_line.set_xdata(racing_line['t'])
    ay_line.set_ydata(racing_line['ay'])

    fig.canvas.draw()
    fig.canvas.flush_events()

    # next state assuming perfect tracking of the racing line and 100ms calculation time
    calc_time = 0.1

    # ensure monotonically increasing s vector
    s_array_unwrap = np.unwrap(
        racing_line['s'],
        discont=track_handler.s[-1] / 2.0,
        period=track_handler.s[-1]
    )
    s = np.interp(calc_time, racing_line['t'], s_array_unwrap) % track_handler.s[-1]
    n = np.interp(calc_time, racing_line['t'], racing_line['n'])
    V = np.interp(calc_time, racing_line['t'], racing_line['V'])
    chi = np.interp(calc_time, racing_line['t'], racing_line['chi'])
    ax = np.interp(calc_time, racing_line['t'], racing_line['ax'])
    ay = np.interp(calc_time, racing_line['t'], racing_line['ay'])

# EOF
