import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
mpl.rcParams['lines.linewidth'] = 2


class Visualizer():

    def __init__(
            self,
            track_handler,
            gg_handler,
            vehicle_params,
            threeD: bool = False,
    ):
        self.track_handler = track_handler
        self.gg_handler = gg_handler
        self.vehicle_params = vehicle_params
        self.threeD = threeD

        plt.ion()
        self.fig = plt.figure()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], figure=self.fig)

        axis_track = plt.subplot(gs[:, 0], projection='3d' if threeD else None)
        axis_track.grid()
        # plot track bounds
        left, right = track_handler.get_track_bounds()
        if self.threeD:
            axis_track.plot(left[0], left[1], left[2], color='k')
            axis_track.plot(right[0], right[1], right[2], color='k')
            axis_track.set_box_aspect((np.ptp(left[0]), np.ptp(left[1]), np.ptp(left[2])))
            self.rl_line, = axis_track.plot3D([0], [0], [0], color='r')
        else:
            axis_track.plot(left[0], left[1], color='k')
            axis_track.plot(right[0], right[1], color='k')
            axis_track.set_aspect('equal')
            self.rl_line, = axis_track.plot([0], [0], color='r')


        self.axis_traj = plt.subplot(gs[0, 1])
        self.axis_traj.set_xlabel("t")
        self.axis_traj.grid()
        self.axis_traj.set_xlim([0.0, 1.0])
        self.axis_traj.set_ylim([0.0, gg_handler.V_max])
        self.V_line, = self.axis_traj.plot([0], [0], color='r', label='$V$')
        self.ax_line, = self.axis_traj.plot([0], [0], color='b', label='$\hat{a}_\mathrm{x}$')
        self.ay_line, = self.axis_traj.plot([0], [0], color='g', label='$\hat{a}_\mathrm{y}$')
        self.axis_traj.legend()

        axis_gg = plt.subplot(gs[1, 1])
        axis_gg.set_title('gg-Diagram')
        axis_gg.set_xlabel(r"$\tilde{a}_\mathrm{y}$")
        axis_gg.set_ylabel(r"$\tilde{a}_\mathrm{x}$")
        self.diamond_line, = axis_gg.plot([0], [0], color='g', label='Diamond-shaped underapproximation')
        self.rho_line, = axis_gg.plot([0], [0], color='b', alpha=0.5, label='Polar coordinates (exact)', linestyle='--')
        self.axy_marker, = axis_gg.plot([0], [0], color='r', marker='o', markersize=10)
        axis_gg.grid()
        axis_gg.set_xlim([-40, 40])
        axis_gg.set_ylim([-40, 20])
        axis_gg.legend(loc='lower right')
        self.V_text = axis_gg.text(-35.0, 15.0, "")
        self.gt_text = axis_gg.text(-35.0, 10.0, "")

    def update(
            self,
            s,
            V,
            n,
            chi,
            ax,
            ay,
            racing_line
    ):
        ax_tilde, ay_tilde, g_tilde = self.track_handler.calc_apparent_accelerations(
            V=max(V, 1.0),
            n=n,
            chi=chi,
            ax=ax,
            ay=ay,
            s=s,
            h=self.vehicle_params['h'],
        )
        self.gt_text.set_text(fr'$\tilde{{g}}={float(g_tilde):.2f}$')
        self.V_text.set_text(fr'$V={V:.2f}$')

        # plot diamond shape for current driving state
        acc_max = self.gg_handler.acc_interpolator(ca.vertcat(V, g_tilde))
        gg_exponent = float(acc_max[0])
        ax_min = float(acc_max[1])
        ax_max = float(acc_max[2])
        ay_max = float(acc_max[3])
        ay_array = np.linspace(-ay_max, ay_max, 100)
        ax_array = - ax_min * np.power(
            1.0 - np.power(np.abs(ay_array) / ay_max, gg_exponent),
            1.0 / gg_exponent,
        )
        self.diamond_line.set_xdata(list(ay_array) + [None] + list(ay_array))
        self.diamond_line.set_ydata(list(np.minimum(ax_array, ax_max)) + [None] + list(-ax_array))

        # plot exact shape of gg-diagram for current driving state
        alpha_array = np.linspace(-np.pi, np.pi, 100)
        rho_array = self.gg_handler.rho_interpolator(np.array(
            [V * np.ones_like(alpha_array), float(g_tilde) * np.ones_like(alpha_array), alpha_array])).full().squeeze()
        self.rho_line.set_xdata([np.cos(alpha_array) * rho_array])
        self.rho_line.set_ydata([np.sin(alpha_array) * rho_array])

        self.axy_marker.set_xdata([float(ay_tilde)])
        self.axy_marker.set_ydata([float(ax_tilde)])

        self.axis_traj.set_xlim([0.0, racing_line['t'][-1] * 1.05])
        self.axis_traj.set_ylim([
            min(0.0, 1.05 * np.min(np.concatenate((racing_line['ax'], racing_line['ay'])))),
            1.05 * np.max(np.concatenate((racing_line['V'], racing_line['ax'], racing_line['ay'])))
        ])
        self.rl_line.set_xdata(racing_line['x'])
        self.rl_line.set_ydata(racing_line['y'])
        if self.threeD:
            self.rl_line.set_3d_properties(racing_line['z'])
        self.V_line.set_xdata(racing_line['t'])
        self.V_line.set_ydata(racing_line['V'])
        self.ax_line.set_xdata(racing_line['t'])
        self.ax_line.set_ydata(racing_line['ax'])
        self.ay_line.set_xdata(racing_line['t'])
        self.ay_line.set_ydata(racing_line['ay'])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
