import os
import sys
import casadi as ca
import numpy as np
import yaml
import pandas as pd
from matplotlib import pyplot as plt

params = {
    'track_name': 'LVMS_3d_smoothed.csv',
    'raceline_name': 'LVMS_3d_dallaraAV21_timeoptimal.csv',
    # 'track_name': 'mount_panorama_3d_smoothed.csv',
    # 'raceline_name': 'mount_panorama_3d_dallaraAV21_timeoptimal.csv',
    'vehicle_name': 'dallaraAV21',
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + params['vehicle_name'] + '.yml')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams', params['vehicle_name'], 'velocity_frame')
track_path = os.path.join(data_path, 'smoothed_track_data')
racing_line_path = os.path.join(data_path, 'global_racing_lines')
sys.path.append(os.path.join(dir_path, '..', 'src'))

# load vehicle and tire parameters
with open(vehicle_params_path, 'r') as stream:
    params.update(yaml.safe_load(stream))

from track3D import Track3D
from ggManager import GGManager


def visualize_trajectory(track_path, raceline_path):
    track_handler = Track3D(
        path=os.path.join(track_path)
    )

    normal_vector = track_handler.get_normal_vector_numpy(
        theta=track_handler.theta,
        mu=track_handler.mu,
        phi=track_handler.phi
    )

    gg_handler = GGManager(
        gg_path=gg_diagram_path,
    )

    trajectory_data_frame = pd.read_csv(raceline_path, sep=',')
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    v_opt = trajectory_data_frame['v_opt'].to_numpy()
    n_opt = trajectory_data_frame['n_opt'].to_numpy()
    chi_opt = trajectory_data_frame['chi_opt'].to_numpy()
    ax_opt = trajectory_data_frame['ax_opt'].to_numpy()
    ay_opt = trajectory_data_frame['ay_opt'].to_numpy()
    jx_opt = trajectory_data_frame['jx_opt'].to_numpy()
    jy_opt = trajectory_data_frame['jy_opt'].to_numpy()

    ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
        V=v_opt, n=n_opt, chi=chi_opt, ax=ax_opt, ay=ay_opt, s=s_opt, h=params['vehicle_params']['h']
    )

    ax = track_handler.visualize(show=False)
    ax.set_title(f'Racing line {os.path.split(raceline_path)[-1]}')
    ax.plot(track_handler.x + normal_vector[0] * n_opt, track_handler.y + normal_vector[1] * n_opt, track_handler.z + normal_vector[2] * n_opt, color='red')

    fig, ax = plt.subplots(nrows=5, num='Racing line')
    ax[0].grid()
    ax[0].plot(s_opt, v_opt, label=r'$V$')
    ax[0].legend()
    ax[1].grid()
    ax[1].plot(s_opt, n_opt, label=r'$n$')
    ax[1].legend()
    ax[2].grid()
    ax[2].plot(s_opt, chi_opt, label=r'$\hat{\chi}$')
    ax[2].legend()
    ax[3].grid()
    ax[3].plot(s_opt, ax_opt, label=r'$\hat{a}_x$')
    ax[3].plot(s_opt, ay_opt, label=r'$\hat{a}_y$')
    ax[3].legend()
    ax[4].grid()
    ax[4].plot(s_opt, jx_opt, label=r'$\hat{j}_x$')
    ax[4].plot(s_opt, jy_opt, label=r'$\hat{j}_y$')
    ax[4].legend()

    fig, ax = plt.subplots(nrows=3, num='Apparent accelerations')
    ax[0].plot(s_opt, g_tilde, label=r'$\tilde{g}$')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(s_opt, ax_tilde, label=r'$\tilde{a}_\mathrm{x}$')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(s_opt, ay_tilde, label=r'$\tilde{a}_\mathrm{y}$')
    ax[2].grid()
    ax[2].legend()

    # Show plot.
    plt.show()

if __name__ == '__main__':

    track_handler = Track3D(
        path=os.path.join(track_path, params['track_name'])
    )
    visualize_trajectory(
        track_path=os.path.join(track_path, params['track_name']),
        raceline_path=os.path.join(racing_line_path, params['raceline_name'])
    )
