import os
import sys
import yaml
from casadi import *
from calc_max_slip_map import calc_max_slip_map
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import subprocess

# settings
vehicle_name = 'dallaraAV21'

# discretization of total velocity
V_min = 10.0  # in m/s
V_max = 90.0  # in m/s
V_N = 20

# discretization of g tilde
g_earth = 9.81
g_factor_min = 0.5
g_factor_max = 3.5
g_N = 20
g_list = np.linspace(g_earth * g_factor_min, g_earth * g_factor_max, g_N)

# discretization of polar coordinate angle
alpha_list = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 250)  # for NLP
alpha_list_interp = np.linspace(-np.pi, np.pi, 250)  # for outputting gggv diagram in polar coordinates

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + vehicle_name + '.yml')
out_path = os.path.join(data_path, 'ggg_diagrams', vehicle_name)

num_cores = multiprocessing.cpu_count()

# load all parameters
with open(vehicle_params_path, 'r') as stream:
    params = yaml.safe_load(stream)
vehicle_params = params['vehicle_params']
tire_params = params['tire_params']

# calculate maximum slip maps for normal loads
N_list, kappa_max_list, lambda_max_list = calc_max_slip_map(tire_params=tire_params, debug_plots=debug_plots)
kappa_max = interpolant("kappa_max", "bspline", [N_list], np.abs(kappa_max_list))
lambda_max = interpolant("lambda_max", "bspline", [N_list], np.abs(lambda_max_list))

# function to calculate ax, ay points for given absolute velocity and g-force base don two-track model
def calc_gg_points(V, g_force, alpha_list):
    # variables with scaling factors
    a_x_n = MX.sym("a_x_n")  # longitudinal acceleration
    a_x_s = 50.0
    a_x = a_x_n * a_x_s
    a_y_n = MX.sym("a_y_n")  # lateral acceleration
    a_y_s = 50.0
    a_y = a_y_n * a_y_s
    u_n = MX.sym("u_n")  # longitudinal velocity
    u_s = 90.0
    u = u_n * u_s
    v_n = MX.sym("v_n")  # lateral velocity
    v_s = 10.0
    v = v_n * v_s
    omega_z_n = MX.sym("omega_z_n")  # yaw rate
    omega_z_s = 0.5
    omega_z = omega_z_n * omega_z_s
    delta_n = MX.sym("delta_n")  # steering angle
    delta_s = vehicle_params["delta_max"] / 5.0
    delta = delta_n * delta_s
    N_fl_n = MX.sym("N_fl_n")  # normal force tire front left
    N_fr_n = MX.sym("N_fr_n")  # normal force tire front right
    N_rl_n = MX.sym("N_rl_n")  # normal force tire rear left
    N_rr_n = MX.sym("N_rr_n")  # normal force tire rear right
    N_fl_s = N_fr_s = N_rl_s = N_rr_s = tire_params["N_0"] * 4
    N_fl = N_fl_n * N_fl_s
    N_fr = N_fr_n * N_fr_s
    N_rl = N_rl_n * N_rl_s
    N_rr = N_rr_n * N_rr_s
    F_x_n = MX.sym("F_x_n")  # total driving force
    F_x_s = 10000
    F_x = F_x_n * F_x_s
    F_x_fl_n = MX.sym("F_x_fl_n")  # longitudinal force tire front left
    F_x_fr_n = MX.sym("F_x_fr_n")  # longitudinal force tire front right
    F_x_rl_n = MX.sym("F_x_rl_n")  # longitudinal force tire rear left
    F_x_rr_n = MX.sym("F_x_rr_n")  # longitudinal force tire rear right
    F_x_fl_s = F_x_fr_s = F_x_rl_s = F_x_rr_s = F_x_s / 2
    F_x_fl = F_x_fl_n * F_x_fl_s
    F_x_fr = F_x_fr_n * F_x_fr_s
    F_x_rl = F_x_rl_n * F_x_rl_s
    F_x_rr = F_x_rr_n * F_x_rr_s
    F_y_fl_n = MX.sym("F_y_fl_n")  # lateral force tire front left
    F_y_fr_n = MX.sym("F_y_fr_n")  # lateral force tire front right
    F_y_rl_n = MX.sym("F_y_rl_n")  # lateral force tire rear left
    F_y_rr_n = MX.sym("F_y_rr_n")  # lateral force tire rear right
    F_y_fl_s = F_y_fr_s = F_y_rl_s = F_y_rr_s = F_x_s / 2
    F_y_fl = F_y_fl_n * F_y_fl_s
    F_y_fr = F_y_fr_n * F_y_fr_s
    F_y_rl = F_y_rl_n * F_y_rl_s
    F_y_rr = F_y_rr_n * F_y_rr_s
    kappa_fl_n = MX.sym("kappa_fl_n")  # longitudinal slip front left
    kappa_fr_n = MX.sym("kappa_fr_n")  # longitudinal slip front right
    kappa_rl_n = MX.sym("kappa_rl_n")  # longitudinal slip rear left
    kappa_rr_n = MX.sym("kappa_rr_n")  # longitudinal slip rear right
    kappa_fl_s = kappa_fr_s = kappa_rl_s = kappa_rr_s = max(kappa_max_list) / 2
    kappa_fl = kappa_fl_n * kappa_fl_s
    kappa_fr = kappa_fr_n * kappa_fr_s
    kappa_rl = kappa_rl_n * kappa_rl_s
    kappa_rr = kappa_rr_n * kappa_rr_s
    lambda_fl_n = MX.sym("lambda_fl_n")  # lateral slip front left
    lambda_fr_n = MX.sym("lambda_fr_n")  # lateral slip front right
    lambda_rl_n = MX.sym("lambda_rl_n")  # lateral slip rear left
    lambda_rr_n = MX.sym("lambda_rr_n")  # lateral slip rear right
    lambda_fl_s = lambda_fr_s = lambda_rl_s = lambda_rr_s = max(lambda_max_list) / 2
    lambda_fl = lambda_fl_n * lambda_fl_s
    lambda_fr = lambda_fr_n * lambda_fr_s
    lambda_rl = lambda_rl_n * lambda_rl_s
    lambda_rr = lambda_rr_n * lambda_rr_s

    x_n = vertcat(
        a_x_n,
        a_y_n,
        u_n,
        v_n,
        omega_z_n,
        delta_n,
        N_fl_n,
        N_fr_n,
        N_rl_n,
        N_rr_n,
        F_x_n,
        F_x_fl_n,
        F_x_fr_n,
        F_x_rl_n,
        F_x_rr_n,
        F_y_fl_n,
        F_y_fr_n,
        F_y_rl_n,
        F_y_rr_n,
        kappa_fl_n,
        kappa_fr_n,
        kappa_rl_n,
        kappa_rr_n,
        lambda_fl_n,
        lambda_fr_n,
        lambda_rl_n,
        lambda_rr_n,
    )

    x_s = vertcat(
        a_x_s,
        a_y_s,
        u_s,
        v_s,
        omega_z_s,
        delta_s,
        N_fl_s,
        N_fr_s,
        N_rl_s,
        N_rr_s,
        F_x_s,
        F_x_fl_s,
        F_x_fr_s,
        F_x_rl_s,
        F_x_rr_s,
        F_y_fl_s,
        F_y_fr_s,
        F_y_rl_s,
        F_y_rr_s,
        kappa_fl_s,
        kappa_fr_s,
        kappa_rl_s,
        kappa_rr_s,
        lambda_fl_s,
        lambda_fr_s,
        lambda_rl_s,
        lambda_rr_s,
    )

    # distribution of total driving force
    k_t_braking = 1.0 / (1.0 + vehicle_params["gamma"])
    k_t_diff = 1.0 - k_t_braking
    k_t = Function("k_t", [a_x_n], [(1 - k_t_diff) + k_t_diff * (0.5 + 0.5 * tanh(10 * a_x))])
    if debug_plots and verbose_debug_plots:
        plt.figure()
        plt.title(rf"Driving force distribution factor $k_t$")
        x = np.linspace(-10 / a_x_s, 5 / a_x_s, 200)
        y = k_t(x)
        plt.plot(x * a_x_s, np.array(y).flatten())
        plt.xlabel(rf"$a_x$")
        plt.ylabel(rf"$k_t$")

    # aerodynamic forces
    F_D = 0.5 * vehicle_params["rho"] * vehicle_params["C_D_A"] * u**2  # drag force
    F_Lf = 0.5 * vehicle_params["rho"] * vehicle_params["C_Lf_A"] * u**2  # front downforce
    F_Lr = 0.5 * vehicle_params["rho"] * vehicle_params["C_Lr_A"] * u**2  # rear downforce

    #
    df_z_fl = (N_fl - tire_params["N_0"]) / tire_params["N_0"]
    df_z_fr = (N_fr - tire_params["N_0"]) / tire_params["N_0"]
    df_z_rl = (N_rl - tire_params["N_0"]) / tire_params["N_0"]
    df_z_rr = (N_rr - tire_params["N_0"]) / tire_params["N_0"]

    # theoretical slips
    sigma_x_fl = kappa_fl / (1 + kappa_fl)
    sigma_x_fr = kappa_fr / (1 + kappa_fr)
    sigma_x_rl = kappa_rl / (1 + kappa_rl)
    sigma_x_rr = kappa_rr / (1 + kappa_rr)

    sigma_y_fl = tan(lambda_fl) / (1 + kappa_fl)
    sigma_y_fr = tan(lambda_fr) / (1 + kappa_fr)
    sigma_y_rl = tan(lambda_rl) / (1 + kappa_rl)
    sigma_y_rr = tan(lambda_rr) / (1 + kappa_rr)

    sigma_fl = sqrt(sigma_x_fl**2 + sigma_y_fl**2)
    sigma_fr = sqrt(sigma_x_fr**2 + sigma_y_fr**2)
    sigma_rl = sqrt(sigma_x_rl**2 + sigma_y_rl**2)
    sigma_rr = sqrt(sigma_x_rr**2 + sigma_y_rr**2)

    # magic formula coefficients (longitudinal)
    K_x_fl = N_fl * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_fl)
    K_x_fr = N_fr * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_fr)
    K_x_rl = N_rl * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_rl)
    K_x_rr = N_rr * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z_rr)

    D_x_fl = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_fl) * tire_params["lambda_mu_x"]
    D_x_fr = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_fr) * tire_params["lambda_mu_x"]
    D_x_rl = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_rl) * tire_params["lambda_mu_x"]
    D_x_rr = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z_rr) * tire_params["lambda_mu_x"]

    B_x_fl = K_x_fl / (tire_params["p_Cx_1"] * D_x_fl * N_fl)
    B_x_fr = K_x_fr / (tire_params["p_Cx_1"] * D_x_fr * N_fr)
    B_x_rl = K_x_rl / (tire_params["p_Cx_1"] * D_x_rl * N_rl)
    B_x_rr = K_x_rr / (tire_params["p_Cx_1"] * D_x_rr * N_rr)

    # magic formula coefficients (lateral)
    K_y_fl = (
        tire_params["N_0"]
        * tire_params["p_Ky_1"]
        * sin(2 * arctan(N_fl / (tire_params["p_Ky_2"] * tire_params["N_0"])))
    )
    K_y_fr = (
        tire_params["N_0"]
        * tire_params["p_Ky_1"]
        * sin(2 * arctan(N_fr / (tire_params["p_Ky_2"] * tire_params["N_0"])))
    )
    K_y_rl = (
        tire_params["N_0"]
        * tire_params["p_Ky_1"]
        * sin(2 * arctan(N_rl / (tire_params["p_Ky_2"] * tire_params["N_0"])))
    )
    K_y_rr = (
        tire_params["N_0"]
        * tire_params["p_Ky_1"]
        * sin(2 * arctan(N_rr / (tire_params["p_Ky_2"] * tire_params["N_0"])))
    )

    D_y_fl = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_fl) * tire_params["lambda_mu_y"]
    D_y_fr = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_fr) * tire_params["lambda_mu_y"]
    D_y_rl = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_rl) * tire_params["lambda_mu_y"]
    D_y_rr = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z_rr) * tire_params["lambda_mu_y"]

    B_y_fl = K_y_fl / (tire_params["p_Cy_1"] * D_y_fl * N_fl)
    B_y_fr = K_y_fr / (tire_params["p_Cy_1"] * D_y_fr * N_fr)
    B_y_rl = K_y_rl / (tire_params["p_Cy_1"] * D_y_rl * N_rl)
    B_y_rr = K_y_rr / (tire_params["p_Cy_1"] * D_y_rr * N_rr)

    # initial guess (maximum positive ax, ay to zero)
    ax0 = vehicle_params["P_max"] / V / vehicle_params["m"]
    Fx0 = vehicle_params["m"] * ax0
    N_ij0 = tire_params["N_0"]
    x0 = vertcat(
        ax0,
        0.0,
        V,
        0.01,
        0.0,
        0.0,
        N_ij0,
        N_ij0,
        N_ij0,
        N_ij0,
        Fx0,
        0.0,
        0.0,
        Fx0 / 2,
        Fx0 / 2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
        0.01,
        0.01,
        0.01,
    )
    x0_n = x0 / x_s  # normalized initial guess

    ax_points = []
    ay_points = []
    beta_points = []
    x_opt_list = []
    for alpha in alpha_list:
        # constraints
        g = []
        lbg = []
        ubg = []
        g += [alpha - arctan2(a_x, a_y)]
        g += [V - sqrt(u**2 + v**2)]
        lbg += [0, 0]
        ubg += [0, 0]

        # lateral slips
        g += [lambda_fl - delta + (v + omega_z * vehicle_params["a"]) / (u + 0.5 * vehicle_params["T"] * omega_z)]
        g += [lambda_fr - delta + (v + omega_z * vehicle_params["a"]) / (u - 0.5 * vehicle_params["T"] * omega_z)]
        g += [lambda_rl + (v - omega_z * vehicle_params["b"]) / (u + 0.5 * vehicle_params["T"] * omega_z)]
        g += [lambda_rr + (v - omega_z * vehicle_params["b"]) / (u - 0.5 * vehicle_params["T"] * omega_z)]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        # minimum and maximum lateral slip
        g += [
            lambda_fl + lambda_max(N_fl),
            lambda_fr + lambda_max(N_fr),
            lambda_rl + lambda_max(N_rl),
            lambda_rr + lambda_max(N_rr),
        ]
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [np.inf, np.inf, np.inf, np.inf]
        g += [
            lambda_fl - lambda_max(N_fl),
            lambda_fr - lambda_max(N_fr),
            lambda_rl - lambda_max(N_rl),
            lambda_rr - lambda_max(N_rr),
        ]
        lbg += [-np.inf, -np.inf, -np.inf, -np.inf]
        ubg += [0.0, 0.0, 0.0, 0.0]

        # minimum and maximum longitudinal slip
        g += [
            kappa_fl + kappa_max(N_fl),
            kappa_fr + kappa_max(N_fr),
            kappa_rl + kappa_max(N_rl),
            kappa_rr + kappa_max(N_rr),
        ]
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [np.inf, np.inf, np.inf, np.inf]
        g += [
            kappa_fl - kappa_max(N_fl),
            kappa_fr - kappa_max(N_fr),
            kappa_rl - kappa_max(N_rl),
            kappa_rr - kappa_max(N_rr),
        ]
        lbg += [-np.inf, -np.inf, -np.inf, -np.inf]
        ubg += [0.0, 0.0, 0.0, 0.0]

        # magic formula as constraints
        g += [
            F_x_fl
            - N_fl
            * sigma_x_fl
            / sigma_fl
            * D_x_fl
            * sin(
                tire_params["p_Cx_1"]
                * arctan(B_x_fl * sigma_fl - tire_params["p_Ex_1"] * (B_x_fl * sigma_fl - arctan(B_x_fl * sigma_fl)))
            )
        ]
        g += [
            F_x_fr
            - N_fr
            * sigma_x_fr
            / sigma_fr
            * D_x_fr
            * sin(
                tire_params["p_Cx_1"]
                * arctan(B_x_fr * sigma_fr - tire_params["p_Ex_1"] * (B_x_fr * sigma_fr - arctan(B_x_fr * sigma_fr)))
            )
        ]
        g += [
            F_x_rl
            - N_rl
            * sigma_x_rl
            / sigma_rl
            * D_x_rl
            * sin(
                tire_params["p_Cx_1"]
                * arctan(B_x_rl * sigma_rl - tire_params["p_Ex_1"] * (B_x_rl * sigma_rl - arctan(B_x_rl * sigma_rl)))
            )
        ]
        g += [
            F_x_rr
            - N_rr
            * sigma_x_rr
            / sigma_rr
            * D_x_rr
            * sin(
                tire_params["p_Cx_1"]
                * arctan(B_x_rr * sigma_rr - tire_params["p_Ex_1"] * (B_x_rr * sigma_rr - arctan(B_x_rr * sigma_rr)))
            )
        ]

        g += [
            F_y_fl
            - N_fl
            * sigma_y_fl
            / sigma_fl
            * D_y_fl
            * sin(
                tire_params["p_Cy_1"]
                * arctan(B_y_fl * sigma_fl - tire_params["p_Ey_1"] * (B_y_fl * sigma_fl - arctan(B_y_fl * sigma_fl)))
            )
        ]
        g += [
            F_y_fr
            - N_fr
            * sigma_y_fr
            / sigma_fr
            * D_y_fr
            * sin(
                tire_params["p_Cy_1"]
                * arctan(B_y_fr * sigma_fr - tire_params["p_Ey_1"] * (B_y_fr * sigma_fr - arctan(B_y_fr * sigma_fr)))
            )
        ]
        g += [
            F_y_rl
            - N_rl
            * sigma_y_rl
            / sigma_rl
            * D_y_rl
            * sin(
                tire_params["p_Cy_1"]
                * arctan(B_y_rl * sigma_rl - tire_params["p_Ey_1"] * (B_y_rl * sigma_rl - arctan(B_y_rl * sigma_rl)))
            )
        ]
        g += [
            F_y_rr
            - N_rr
            * sigma_y_rr
            / sigma_rr
            * D_y_rr
            * sin(
                tire_params["p_Cy_1"]
                * arctan(B_y_rr * sigma_rr - tire_params["p_Ey_1"] * (B_y_rr * sigma_rr - arctan(B_y_rr * sigma_rr)))
            )
        ]

        lbg += [0, 0, 0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0, 0, 0]

        # roll stiffness balance
        g += [
            vehicle_params["m"] * a_y * vehicle_params["h"] / vehicle_params["T"] * vehicle_params["epsilon"]
            - 0.5 * (N_fl - N_fr)
        ]
        lbg += [0]
        ubg += [0]

        # open-differential assumption
        g += [F_x_fl - 0.5 * (1.0 - k_t(a_x)) * F_x]
        g += [F_x_fr - 0.5 * (1.0 - k_t(a_x)) * F_x]
        g += [F_x_rl - 0.5 * k_t(a_x) * F_x]
        g += [F_x_rr - 0.5 * k_t(a_x) * F_x]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        # steady state equations
        g += [vehicle_params["m"] * a_x - (F_x_fl + F_x_fr + F_x_rl + F_x_rr) + (F_y_fl + F_y_fr) * delta + F_D]
        g += [vehicle_params["m"] * a_y - (F_y_fl + F_y_fr + F_y_rl + F_y_rr) - (F_x_fl + F_x_fr) * delta]
        g += [vehicle_params["m"] * g_force + F_Lf + F_Lr - N_fl - N_fr - N_rl - N_rr]
        g += [vehicle_params["m"] * a_y * vehicle_params["h"] - 0.5 * vehicle_params["T"] * (N_fl - N_fr + N_rl - N_rr)]
        g += [
            vehicle_params["m"] * a_x * vehicle_params["h"]
            - vehicle_params["a"] * F_Lf
            + vehicle_params["b"] * F_Lr
            + vehicle_params["a"] * (N_fl + N_fr)
            - vehicle_params["b"] * (N_rl + N_rr)
        ]
        g += [
            0.5 * vehicle_params["T"] * (F_y_fl - F_y_fr) * delta
            - vehicle_params["a"] * (F_x_fl + F_x_fr) * delta
            + 0.5 * vehicle_params["T"] * (-F_x_fl + F_x_fr - F_x_rl + F_x_rr)
            - vehicle_params["a"] * (F_y_fl + F_y_fr)
            + vehicle_params["b"] * (F_y_rl + F_y_rr)
        ]
        g += [a_y - omega_z * u]

        lbg += [0, 0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0, 0]

        # engine limits
        g += [F_x * u]
        lbg += [-np.inf]
        ubg += [vehicle_params["P_max"]]

        # velocity limit
        g += [u]
        lbg += [0.0]
        ubg += [vehicle_params['v_max']]

        # steering angle limit
        g += [delta]
        lbg += [-vehicle_params["delta_max"]]
        ubg += [vehicle_params["delta_max"]]

        # positive normal forces
        g += [N_fl, N_fr, N_rl, N_rr]
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [tire_params["N_max"], tire_params["N_max"], tire_params["N_max"], tire_params["N_max"]]

        # cost function
        f = - a_x**2 - a_y**2  # maximize substitute radius

        # NLP solver
        nlp = {"x": x_n, "f": f, "g": vertcat(*g)}
        opts = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        solver = nlpsol("solver", "ipopt", nlp, opts)

        x_opt = solver(x0=x0_n, lbx=-np.inf, ubx=np.inf, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        if solver.stats()["success"]:
            x0 = x_opt["x"]  # new initial guess
            ax_points.append(x_opt["x"][0] * a_x_s)
            ay_points.append(x_opt["x"][1] * a_y_s)
            beta_points.append(arctan(abs(x_opt["x"][3] * v_s) / abs(x_opt["x"][2] * u_s)))
            x_opt_list.append(np.array(x0))
        else:
            pass  # print(f"No solution for V={V}, g={g_force}, alpha={alpha}.")

    return np.array(ax_points), np.array(ay_points), np.array(beta_points)


# generate map of acceleration limits in polar coordinates with substitute radius for each polar angle
def gen_gg_polar(ax_ay_pairs, alpha_list):
    # mirror ax ay pairs
    ax_neg_ay_pairs = ax_ay_pairs.copy()
    ax_neg_ay_pairs[:, 1] = -ax_neg_ay_pairs[:, 1]
    ax_ay_pairs_mirrored = np.row_stack((ax_ay_pairs, ax_neg_ay_pairs[::-1]))

    for _ in range(1):
        alpha_points = []
        ax_points = []
        ay_points = []
        for ax, ay in zip(ax_ay_pairs_mirrored[:, 0], ax_ay_pairs_mirrored[:, 1]):
            angle = np.arctan2(ax, ay)
            alpha_points.append(angle)
            ax_points.append(ax)
            ay_points.append(ay)

        ax_points_interp = np.interp(alpha_list, alpha_points, ax_points, period=2 * np.pi)
        ay_points_interp = np.interp(alpha_list, alpha_points, ay_points, period=2 * np.pi)

        ax_ay_pairs_mirrored = np.column_stack((ax_points_interp, ay_points_interp))

    rho_list = np.sqrt(ax_ay_pairs_mirrored[:, 0] ** 2 + ax_ay_pairs_mirrored[:, 1] ** 2)
    return rho_list

# rotate ax ay pairs from vehicle frame to velocity-aligned frame based on the side-slip angle
def rotate_by_beta(ax_points_vehicle_frame, ay_points_vehicle_frame, beta_points):
    ax_points_velocity_frame = (
        np.cos(beta_points) * np.array(ax_points_vehicle_frame).squeeze()
        + np.sin(beta_points) * np.array(ay_points_vehicle_frame).squeeze()
    )
    ay_points_velocity_frame = (
        -np.sin(beta_points) * np.array(ax_points_vehicle_frame).squeeze()
        + np.cos(beta_points) * np.array(ay_points_vehicle_frame).squeeze()
    )
    return ax_points_velocity_frame, ay_points_velocity_frame


def calc_rho_for_V(V):
    print(f'Start calculation for V={V} ...')
    rho_vehicle_frame_tmp = []

    rho_velocity_frame_tmp = []
    for g_i, g_force in enumerate(g_list):
        ax_points_vehicle_frame, ay_points_vehicle_frame, beta_points = calc_gg_points(V, g_force, alpha_list)

        # rotate for velocity frame
        ax_points_velocity_frame, ay_points_velocity_frame = rotate_by_beta(
            ax_points_vehicle_frame, ay_points_vehicle_frame, beta_points
        )

        # rho vehicle frame
        rho_list_vehicle_frame = gen_gg_polar(
            np.column_stack((ax_points_vehicle_frame, ay_points_vehicle_frame)).squeeze(), alpha_list_interp
        )
        rho_vehicle_frame_tmp.append(rho_list_vehicle_frame)

        # rho velocity frame
        rho_list_velocity_frame = gen_gg_polar(
            np.column_stack((ax_points_velocity_frame, ay_points_velocity_frame)).squeeze(), alpha_list_interp
        )
        rho_velocity_frame_tmp.append(rho_list_velocity_frame)

    print(f'... Finished calculation for V={V}')
    return rho_vehicle_frame_tmp, rho_velocity_frame_tmp


if __name__ == "__main__":
    v_max_found = False
    V_list = np.linspace(V_min, V_max, V_N)

    processed_list = Parallel(
        n_jobs=num_cores
    )(
        delayed(calc_rho_for_V)(V) for V in V_list
    )

    rho_vehicle_frame = [tmp[0] for tmp in processed_list]
    rho_velocity_frame = [tmp[1] for tmp in processed_list]

    for frame in ["vehicle_frame", "velocity_frame"]:
        os.makedirs(os.path.join(out_path, frame), exist_ok=True)
        np.save(os.path.join(out_path, frame, "v_list.npy"), V_list)
        np.save(os.path.join(out_path, frame, "g_list.npy"), g_list)
        # interpolated values
        np.save(os.path.join(out_path, frame, "alpha_list.npy"), alpha_list_interp)
        np.save(
            os.path.join(out_path, frame, "rho.npy"),
            np.asarray(rho_vehicle_frame) if frame == "vehicle_frame" else np.asarray(rho_velocity_frame),
        )

# EOF
