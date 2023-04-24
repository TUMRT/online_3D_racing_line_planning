import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import casadi as ca

vehicle_name = 'dallaraAV21'
frame = 'velocity'
# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
gg_path = os.path.join(data_path, 'gg_diagrams', vehicle_name, frame+'_frame')
sys.path.append(os.path.join(dir_path, '..', 'src'))

from ggManager import GGManager

# casadi interpolator
gg_handler = GGManager(
    gg_path=gg_path,
    gg_margin=0.0,
)

v_test = np.linspace(0.0, gg_handler.V_max, 5)
g_test = np.linspace(0.0, gg_handler.g_max, 100)
alpha_test = np.linspace(-np.pi, np.pi, 300)
g = 9.81
# g = 0.5 * 9.81
# g = 2.0 * 9.81

fig_polar = plt.figure('Polar coordinates')
ax_polar = fig_polar.add_subplot(projection='3d')
ax_polar.set_title(f'Polar coordinates')

fig_form = plt.figure('Diamond shpaed')
ax_diamond = fig_form.add_subplot()
ax_diamond.set_title(f'Diamond shaped underapproximations')

X_test, Y_test = np.meshgrid(g_test, alpha_test, indexing='ij')
for V in v_test:
    Z_test = np.zeros_like(X_test)
    for g_i, gt in enumerate(g_test):
        for alpha_i, alpha in enumerate(alpha_test):
            Z_test[g_i, alpha_i] = gg_handler.rho_interpolator_no_margin([V, gt, alpha])
    ax_polar.plot_surface(Y_test, X_test, Z_test)

    gg_exponent = float(gg_handler.gg_exponent_interpolator(ca.vertcat(V, g)))
    ax_max = float(gg_handler.ax_max_interpolator(ca.vertcat(V, g)))
    ax_min = float(gg_handler.ax_min_interpolator(ca.vertcat(V, g)))
    ay_max = float(gg_handler.ay_max_interpolator(ca.vertcat(V, g)))
    # Polar plot
    rho_test = gg_handler.rho_interpolator_no_margin(np.array([V * np.ones_like(alpha_test), g * np.ones_like(alpha_test), alpha_test])).full().squeeze()

    # Form shape
    tmp = ax_diamond.plot(np.cos(alpha_test) * rho_test, np.sin(alpha_test) * rho_test, label=f'$V$={V}, $\\tilde{{g}}$={g}', alpha=0.3)
    ay_array = np.linspace(-ay_max, ay_max, 200)
    ax_array = np.zeros_like(ay_array)
    for i, ay in enumerate(ay_array):
        ax_array[i] = - ax_min * np.power(
            1.0 - np.power(np.abs(ay) / ay_max, gg_exponent),
            1.0 / gg_exponent,
        )
    ax_diamond.plot(ay_array, np.minimum(ax_array, ax_max), color=tmp[0].get_color())
    ax_diamond.plot(ay_array, -ax_array, color=tmp[0].get_color())

ax_polar.set_xlabel(r'$\alpha$')
ax_polar.set_ylabel(r'$\tilde{g}$')
ax_polar.set_zlabel(r'$\rho$')

ax_diamond.set_xlabel(r'$\tilde{a}_\mathrm{y}$')
ax_diamond.set_ylabel(r'$\tilde{a}_\mathrm{x}$')
ax_diamond.legend()
ax_diamond.set_aspect('equal')

plt.show()

# EOF
