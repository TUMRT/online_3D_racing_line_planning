import os
from casadi import *
import multiprocessing
from joblib import Parallel, delayed

vehicle_name = 'dallaraAV21'
# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams')

num_cores = multiprocessing.cpu_count()


def gen_diamond_representation(alpha_list, rho_list):
    # diamond representation
    gg_exponent = MX.sym('gg_exponent')
    ax_min = MX.sym('ax_min')
    ax_max = MX.sym('ax_max')
    ay_max = MX.sym('ay_max')

    x = vertcat(
        gg_exponent,
        ax_min,
        ax_max,
        ay_max
    )
    lbx = vertcat(
        1.0,
        - np.interp(-np.pi / 2.0, alpha_list, rho_list),
        0.0,
        0.0
    )
    ubx = vertcat(
        2.0,
        0.0,
        np.interp(np.pi / 2.0, alpha_list, rho_list),
        np.interp(0.0, alpha_list, rho_list, 0.0)
    )

    f = 0
    g_const = []
    lbg = []
    ubg = []

    for alpha in np.linspace(
            -np.pi / 2.0,
            np.pi / 2.0,
            200,
    ):
        rho_max = np.interp(alpha, alpha_list, rho_list)

        ay = power(
            fabs(ax_min) ** gg_exponent / (
                        tan(fabs(alpha)) ** gg_exponent + (fabs(ax_min) / ay_max) ** gg_exponent),
            1.0 / gg_exponent
        )
        ax = ay * tan(fabs(alpha))
        rho_diamond = sqrt(ax ** 2 + ay ** 2)

        if alpha > 0.0:
            rho_diamond = fmin(
                rho_diamond,
                ax_max / sin(alpha)
            )

        g_const += [rho_max - rho_diamond]
        lbg += [0.0]
        ubg += [np.inf]

        f -= rho_diamond**2

    x0 = vertcat(
        1.0,
        -5.1,
        5.0,
        5.0
    )
    nlp = {"x": x, "f": f, "g": vertcat(*g_const)}
    opts = {
        "verbose": False, "ipopt.print_level": 0, "print_time": 0,
        "ipopt.hessian_approximation": 'limited-memory',
    }
    solver = nlpsol("solver", "ipopt", nlp, opts)
    x_opt = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=vertcat(*lbg), ubg=vertcat(*ubg))['x']

    gg_exponent = float(x_opt[0])
    ax_min = float(x_opt[1])
    ax_max = float(x_opt[2])
    ay_max = float(x_opt[3])

    return gg_exponent, ax_min, ax_max, ay_max


def gen_diamond_representation_for_V(alpha_list, rho_list):
    gg_exponent_tmp = []
    ax_min_tmp = []
    ax_max_tmp = []
    ay_max_tmp = []

    for rho in rho_list:
        gg_exponent, ax_min, ax_max, ay_max = gen_diamond_representation(
            alpha_list=alpha_list,
            rho_list=rho
        )
        gg_exponent_tmp.append(gg_exponent)
        ax_min_tmp.append(ax_min)
        ax_max_tmp.append(ax_max)
        ay_max_tmp.append(ay_max)

    return gg_exponent_tmp, ax_min_tmp, ax_max_tmp, ay_max_tmp


for frame in ['vehicle', 'velocity']:
    path = os.path.join(gg_diagram_path, vehicle_name, frame + '_frame')
    V_list = np.load(os.path.join(path, 'v_list.npy'))
    V_max = V_list.max()
    g_list = np.load(os.path.join(path, 'g_list.npy'))
    g_min = g_list.min()
    g_max = g_list.max()
    # polar coordinates
    alpha_list = np.load(os.path.join(path, 'alpha_list.npy'))
    rho_list = np.load(os.path.join(path, 'rho.npy'))

    processed_list = Parallel(
        n_jobs=num_cores
    )(
        delayed(gen_diamond_representation_for_V)(alpha_list, rho) for rho in rho_list
    )

    gg_exponent_list = [tmp[0] for tmp in processed_list]
    ax_min_list = [tmp[1] for tmp in processed_list]
    ax_max_list = [tmp[2] for tmp in processed_list]
    ay_max_list = [tmp[3] for tmp in processed_list]

    out_path = os.path.join(gg_diagram_path, vehicle_name)
    np.save(os.path.join(out_path, frame + '_frame', "gg_exponent.npy"), gg_exponent_list)
    np.save(os.path.join(out_path, frame + '_frame', "ax_min.npy"), ax_min_list)
    np.save(os.path.join(out_path, frame + '_frame', "ax_max.npy"), ax_max_list)
    np.save(os.path.join(out_path, frame + '_frame', "ay_max.npy"), ay_max_list)

# EOF
