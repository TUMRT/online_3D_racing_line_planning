from acados_template import AcadosModel
import casadi as ca

def export_point_mass_ode_model(
        vehicle_params,
        track_handler,
        gg_handler,
        optimization_horizon: float,
        gg_mode: str = 'polar',
        weight_jerk: float = 1e-2,
        neglect_w_terms: bool = True,
        neglect_euler: bool = True,
        neglect_centrifugal: bool = True,
        neglect_w_dot: bool = False,
        w_slack_V: float = 50.0,
):

    model_name = 'point_mass_ode'

    # Longitudinal position as independent variable
    s = ca.MX.sym('s')
    # Velocity.
    V = ca.MX.sym('V')
    # Lateral displacement.
    n = ca.MX.sym('n')
    # Relative orientation w.r.t. reference line.
    chi = ca.MX.sym('chi')
    # Longitudinal acceleration.
    ax = ca.MX.sym('ax')
    # Lateral acceleration.
    ay = ca.MX.sym('ay')

    # Create state vector.
    x = ca.vertcat(s, V, n, chi, ax, ay)

    # Longitudinal jerk.
    jx = ca.MX.sym('jx')
    # Lateral jerk.
    jy = ca.MX.sym('jy')
    # slacks
    epsilon_V = ca.MX.sym('epsilon_V')
    # Control vector and longitudinal position on track
    u = ca.vertcat(jx, jy, epsilon_V)

    # Derivatives with respect to s (not time!)
    ds = ca.MX.sym('ds')
    dV = ca.MX.sym('dV')
    dn = ca.MX.sym('dn')
    dchi = ca.MX.sym('dchi')
    dax = ca.MX.sym('dax')
    day = ca.MX.sym('day')
    dx = ca.vertcat(ds, dV, dn, dchi, dax, day)

    # Dynamics
    # Time-distance scaling factor (dt/ds).
    s_dot = (V * ca.cos(chi)) / (1.0 - n * track_handler.Omega_z_interpolator(s))
    # vertical velocity
    w = n * track_handler.Omega_x_interpolator(s) * s_dot

    f_ds = 1.0

    f_dV = 1.0 / s_dot * ax
    if not neglect_w_terms:
        f_dV += w * (track_handler.Omega_x_interpolator(s) * ca.sin(chi) - track_handler.Omega_y_interpolator(s) * ca.cos(chi))

    f_dn = 1.0 / s_dot * V * ca.sin(chi)

    f_dchi = 1.0 / s_dot * ay / V - track_handler.Omega_z_interpolator(s)
    if not neglect_w_terms:
        f_dchi += w * (track_handler.Omega_x_interpolator(s) * ca.cos(chi) + track_handler.Omega_y_interpolator(s) * ca.sin(chi)) / V

    f_dax = jx / s_dot
    f_day = jy / s_dot

    # explicit
    f_expl = ca.vertcat(
        f_ds,
        f_dV,
        f_dn,
        f_dchi,
        f_dax,
        f_day
    )
    # implicit
    f_impl = dx - f_expl

    model = AcadosModel()
    model.name = model_name
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = dx
    model.u = u

    # cost for time optimality
    model.cost_expr_ext_cost = 1.0 / s_dot + weight_jerk * (jx / s_dot)**2 + weight_jerk * (jy / s_dot)**2 + \
                               w_slack_V/optimization_horizon*(epsilon_V / s_dot) + w_slack_V/10.0/optimization_horizon*(epsilon_V / s_dot)**2

    # apparent accelerations
    ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
        V=V,
        n=n,
        chi=chi,
        ax=ax,
        ay=ay,
        s=s,
        h=vehicle_params['h'],
        neglect_w_omega_y=neglect_w_terms,
        neglect_w_omega_x=neglect_w_terms,
        neglect_euler=neglect_euler,
        neglect_centrifugal=neglect_centrifugal,
        neglect_w_dot=neglect_w_dot
    )

    # gggv-diagram constraints
    if gg_mode == 'polar':
        rho = ca.sqrt(ax_tilde ** 2 + ay_tilde ** 2)
        alpha = ca.arctan2(ax_tilde, ay_tilde)
        adherence_radius = gg_handler.rho_interpolator(ca.vertcat(V, g_tilde, alpha))
        model.con_h_expr = ca.vertcat(
            adherence_radius - rho,
        )
    elif gg_mode == 'diamond':
        acc_max = gg_handler.acc_interpolator(ca.vertcat(V, g_tilde))
        gg_exponent = acc_max [0]
        ax_min = acc_max[1]
        ax_max = acc_max[2]
        ay_max = acc_max[3]
        model.con_h_expr = ca.vertcat(
            ax_max - ax_tilde,
            ay_max - ca.fabs(ay_tilde),
            ca.fabs(ax_min) * ca.power(
                ca.fmax(
                    (1.0 - ca.power(
                        ca.fmin(ca.fabs(ay_tilde) / ay_max, 1.0),
                        gg_exponent
                    ))
                    , 1e-3
                ),
                1.0 / gg_exponent
            ) - ca.fabs(ax_tilde)
        )

    return model

# EOF
