import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import casadi as ca


vehicle_name = 'dallaraAV21'
mode = 'velocity'
# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', '..', '..', 'data')
ggg_diagram_path_vehicle = os.path.join(data_path, 'ggg_diagrams', vehicle_name, 'vehicle_frame')
ggg_diagram_path_velocity = os.path.join(data_path, 'ggg_diagrams', vehicle_name, 'velocity_frame')
sys.path.append(os.path.join(dir_path, '..', '..', '..', 'track'))

from gggvManager import GGGVManager

# casadi interpolator
gggv_handler = GGGVManager(
    gggv_path=ggg_diagram_path_velocity if mode=='velocity' else ggg_diagram_path_vehicle,
    gg_margin=0.0,
)

v_test = np.linspace(0.0, gggv_handler.V_max, 5)
g_test = np.linspace(0.0, gggv_handler.g_max, 100)
alpha_test = np.linspace(-np.pi, np.pi, 300)

fig_polar = plt.figure(1)
ax_polar = fig_polar.add_subplot(projection='3d')
ax_polar.set_title(f'Polar {mode} frame')

fig_form = plt.figure(4)
ax_form = fig_form.add_subplot()
ax_form.set_title(f'Form {mode} frame')

X_test, Y_test = np.meshgrid(g_test, alpha_test, indexing='ij')
for V in v_test:
    Z_test = np.zeros_like(X_test)
    for g_i, g in enumerate(g_test):
        for alpha_i, alpha in enumerate(alpha_test):
            Z_test[g_i, alpha_i] = gggv_handler.gggv_interpolator_no_margin([V, g, alpha])
    ax_polar.plot_surface(Y_test, X_test, Z_test)

    g = 9.81
    gg_exponent = float(gggv_handler.gg_exponent_interpolator(ca.vertcat(V, g)))
    ax_max = float(gggv_handler.ax_max_interpolator(ca.vertcat(V, g)))
    ax_min = float(gggv_handler.ax_min_interpolator(ca.vertcat(V, g)))
    ay_max = float(gggv_handler.ay_max_interpolator(ca.vertcat(V, g)))
    # Polar plot
    rho_test = gggv_handler.gggv_interpolator_no_margin(np.array([V * np.ones_like(alpha_test), g * np.ones_like(alpha_test), alpha_test])).full().squeeze()

    # Form shape
    tmp = ax_form.plot(np.cos(alpha_test) * rho_test, np.sin(alpha_test) * rho_test, label=f'V={V}, g={g}', alpha=0.3)
    ay_array = np.linspace(-ay_max, ay_max, 200)
    ax_array = np.zeros_like(ay_array)
    for i, ay in enumerate(ay_array):
        ax_array[i] = - ax_min * np.power(
            1.0 - np.power(np.abs(ay) / ay_max, gg_exponent),
            1.0 / gg_exponent,
        )
    ax_form.plot(ay_array, np.minimum(ax_array, ax_max), color=tmp[0].get_color())
    ax_form.plot(ay_array, -ax_array, color=tmp[0].get_color())

ax_polar.set_xlabel('alpha')
ax_polar.set_ylabel('g-force')
ax_polar.set_zlabel(r'$\rho$')

ax_form.set_xlabel(r'$a_\mathrm{y}$')
ax_form.set_ylabel(r'$a_\mathrm{x}$')
ax_form.legend()
ax_form.set_aspect('equal')

plt.show()