from casadi import *
from tqdm import tqdm


def calc_max_slip_map(tire_params: dict, debug_plots: bool = False):
    N_list = np.linspace(10.0, tire_params["N_max"], 100)
    N_list_res = []
    kappa_max_list = []
    lambda_max_list = []
    F_x_list = []
    F_y_list = []

    kappa = MX.sym("kappa")  # longitudinal slip
    lambdaa = MX.sym("lambda")  # lateral slip
    F_x = MX.sym("F_x")
    F_y = MX.sym("F_y")

    x0_x = [0.1, 0, tire_params["N_0"], 0]
    x0_y = [0, 0.1, 0, -tire_params["N_0"]]

    print(f"Calculating maximum slips for normal tire loads:\n")
    for i, N in tqdm(enumerate(N_list), total=len(N_list)):

        df_z = (N - tire_params["N_0"]) / tire_params["N_0"]

        sigma_x = kappa / (1 + kappa)
        sigma_y = tan(lambdaa) / (1 + kappa)
        sigma = sqrt(sigma_x**2 + sigma_y**2)

        K_x = N * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z)
        D_x = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z) * tire_params["lambda_mu_x"]
        B_x = K_x / (tire_params["p_Cx_1"] * D_x * N)

        K_y = (
            tire_params["N_0"]
            * tire_params["p_Ky_1"]
            * sin(2 * arctan(N / (tire_params["p_Ky_2"] * tire_params["N_0"])))
        )
        D_y = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z) * tire_params["lambda_mu_y"]
        B_y = K_y / (tire_params["p_Cy_1"] * D_y * N)

        # constraints
        g = []
        lbg = []
        ubg = []
        g += [
            F_x
            - N
            * sigma_x
            / sigma
            * D_x
            * sin(
                tire_params["p_Cx_1"]
                * arctan(B_x * sigma - tire_params["p_Ex_1"] * (B_x * sigma - arctan(B_x * sigma)))
            )
        ]
        g += [
            F_y
            - N
            * sigma_y
            / sigma
            * D_y
            * sin(
                tire_params["p_Cy_1"]
                * arctan(B_y * sigma - tire_params["p_Ey_1"] * (B_y * sigma - arctan(B_y * sigma)))
            )
        ]
        lbg += [0, 0]
        ubg += [0, 0]

        # cost function
        f_x = -F_x
        f_y = F_y

        # NLP
        x = vertcat(kappa, lambdaa, F_x, F_y)
        lbx = vertcat(0, 0, 0, -np.inf)
        ubx = vertcat(tire_params["kappa_max"], tire_params["lambda_max"], np.inf, 0)
        nlp_x = {"x": x, "f": f_x, "g": vertcat(*g)}
        nlp_y = {"x": x, "f": f_y, "g": vertcat(*g)}

        opts = {"ipopt.print_level": 0, "print_time": 0}
        solver_x = nlpsol("solver_x", "ipopt", nlp_x, opts)
        solver_y = nlpsol("solver_y", "ipopt", nlp_y, opts)

        x_opt_x = solver_x(x0=x0_x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_opt_y = solver_y(x0=x0_y, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        if solver_x.stats()["success"] and solver_y.stats()["success"]:  # only if solver converges
            N_list_res.append(N)
            kappa_max_list.append(float(x_opt_x["x"][0]))
            F_x_list.append(float(x_opt_x["x"][2]))
            lambda_max_list.append(float(x_opt_y["x"][1]))
            F_y_list.append(float(x_opt_y["x"][3]))

    if debug_plots:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=2)
        ax[0].set_title("Normal force dependent maximal slip")
        ax[0].plot(N_list_res, kappa_max_list, label=rf"$\kappa$ (longitudinal)", marker='o')
        ax[0].plot(N_list_res, lambda_max_list, label=rf"$\lambda$ (lateral)", marker='o')
        ax[0].legend()
        ax[1].set_title("Tire forces")
        ax[1].plot(N_list_res, np.abs(F_x_list), label=rf"$F_x$", marker='o')
        ax[1].plot(N_list_res, np.abs(F_y_list), label=rf"$F_y$", marker='o')
        ax[1].legend()
        plt.show()

    return N_list_res, kappa_max_list, lambda_max_list

# EOF
