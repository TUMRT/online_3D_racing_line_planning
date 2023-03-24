import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver


class LocalRacinglinePlanner():

    def __init__(
            self,
            params,
            track_handler,
            gg_handler,
            model,
            nlp_solver_type: str = 'SQP',
            qp_solver: str = 'PARTIAL_CONDENSING_HPIPM',
            optimization_horizon: float = 300.0,
            gg_mode: str = 'diamond',
            N_steps: int = 150,
            qp_max_iter: int = 100,
            sqp_max_iter: int = 20,
            globalization: str = 'FIXED_STEP',
            step_length: float = 1.0,
            alpha_reduction: float = 0.7,
            alpha_min: float = 0.05,
            w_slack_n: float = 1e0,
            w_slack_gg: float = 1e0,
            nlp_solver_tol_stat=1e-2,
            nlp_solver_tol_eq=1e-2,
            nlp_solver_tol_ineq=5e-2,
            nlp_solver_tol_comp=1e-2,
            qp_solver_tol_stat=1e-3,
            qp_solver_tol_eq=1e-3,
            qp_solver_tol_ineq=1e-3,
            qp_solver_tol_comp=1e-3,
            neglect_w_terms: bool = True,
            neglect_euler: bool = True,
            neglect_centrifugal: bool = True,
            neglect_w_dot: bool = False,
            neglect_V_omega: bool = False,
    ):
        self.vehicle_params = params['vehicle_params']
        self.tire_params = params['tire_params']
        self.track_handler = track_handler
        self.gg_handler = gg_handler
        self.gg_mode = gg_mode
        self.model = model

        self.N_steps = N_steps
        self.optimization_horizon = optimization_horizon
        self.neglect_w_terms = neglect_w_terms
        self.neglect_euler = neglect_euler
        self.neglect_centrifugal = neglect_centrifugal
        self.neglect_w_dot = neglect_w_dot
        self.neglect_V_omega = neglect_V_omega

        self.initialize_solver(
            nlp_solver_type=nlp_solver_type,
            qp_solver=qp_solver,
            qp_max_iter=qp_max_iter,
            sqp_max_iter=sqp_max_iter,
            globalization=globalization,
            step_length=step_length,
            alpha_reduction=alpha_reduction,
            alpha_min=alpha_min,
            w_slack_n=w_slack_n,
            w_slack_gg=w_slack_gg,
            nlp_solver_tol_stat=nlp_solver_tol_stat,
            nlp_solver_tol_eq=nlp_solver_tol_eq,
            nlp_solver_tol_ineq=nlp_solver_tol_ineq,
            nlp_solver_tol_comp=nlp_solver_tol_comp,
            qp_solver_tol_stat=qp_solver_tol_stat,
            qp_solver_tol_eq=qp_solver_tol_eq,
            qp_solver_tol_ineq=qp_solver_tol_ineq,
            qp_solver_tol_comp=qp_solver_tol_comp,
        )

    def initialize_solver(
            self,
            nlp_solver_type: str,
            qp_solver: str,
            qp_max_iter: float,
            sqp_max_iter: float,
            globalization: str,
            step_length: float,
            alpha_reduction: float,
            alpha_min: float,
            w_slack_n: float = 0.25,
            w_slack_gg: float = 0.1,
            nlp_solver_tol_stat=1e-3,
            nlp_solver_tol_eq=1e-3,
            nlp_solver_tol_ineq=5e-3,
            nlp_solver_tol_comp=1e-3,
            qp_solver_tol_stat=1e-3,
            qp_solver_tol_eq=1e-3,
            qp_solver_tol_ineq=1e-3,
            qp_solver_tol_comp=1e-3,
    ):
        # OCP
        self.ocp = AcadosOcp()
        self.ocp.dims.N = self.N_steps
        self.ocp.model = self.model

        # cost
        self.ocp.cost.cost_type = 'EXTERNAL'

        # constraints
        self.ocp.constraints.x0 = np.zeros(6)

        # constraints on u
        self.ocp.constraints.idxbu = np.array([2])  # indices for epsilon_V
        self.ocp.constraints.lbu = np.array([0.0])
        self.ocp.constraints.ubu = np.array([0.0])

        # constraints on x
        self.ocp.constraints.idxbx = np.array([2, 3])  # indices for n, chi
        self.ocp.constraints.lbx = np.array([-100.0, - np.pi / 2.0])
        self.ocp.constraints.ubx = np.array([100.0, np.pi / 2.0])

        # slacks on x
        self.ocp.constraints.Jsbx = np.array(
            [[1.0],
             [0.0]]
        )  # only n (track boundary is soft constraint)

        # polytopic constraints
        self.ocp.constraints.C = np.array(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]  # V - epsilon_V
        )
        self.ocp.constraints.D = np.array(
            [[0.0, 0.0, -1.0]]  # V - epsilon_V
        )
        self.ocp.constraints.lg = np.array(
            [0.0]
        )
        self.ocp.constraints.ug = np.array(
            [100.0]
        )

        # nonlinear constraints
        if self.gg_mode == 'polar':
            self.ocp.constraints.lh = np.array([
                0.0,  # rho
            ])
            self.ocp.constraints.uh = np.array([
                100.0,  # rho
            ])

            self.ocp.constraints.Jsh = np.array(
                [[1.0]]
            )
            # weights for slack variables (n and rho)
            self.ocp.cost.Zl = np.array([w_slack_n, w_slack_gg])
            self.ocp.cost.zl = self.ocp.cost.Zl / 10.0
            self.ocp.cost.Zu = np.array([w_slack_n, w_slack_gg])
            self.ocp.cost.zu = self.ocp.cost.Zu / 10.0

        elif self.gg_mode == 'diamond':
            self.ocp.constraints.lh = np.array([
                0.0,  # ay
                0.0,  # ax tire
                0.0,  # ax engine
            ])
            self.ocp.constraints.uh = np.array([
                100.0,
                100.0,
                100.0,
            ])
            self.ocp.constraints.Jsh = np.eye(3)

            # weights for slack variables (n, ax, ay, axy)
            self.ocp.cost.Zl = np.array([w_slack_n, w_slack_gg, w_slack_gg, w_slack_gg])
            self.ocp.cost.zl = self.ocp.cost.Zl / 10.0
            self.ocp.cost.Zu = np.array([w_slack_n, w_slack_gg, w_slack_gg, w_slack_gg])
            self.ocp.cost.zu = self.ocp.cost.Zu / 10.0

        # sim and solver settings
        self.ocp.solver_options.qp_solver = qp_solver
        self.ocp.solver_options.nlp_solver_type = nlp_solver_type
        self.ocp.solver_options.regularize_method = "MIRROR"
        self.ocp.solver_options.qp_solver_iter_max = qp_max_iter
        self.ocp.solver_options.nlp_solver_max_iter = sqp_max_iter
        self.ocp.solver_options.globalization = globalization
        self.ocp.solver_options.nlp_solver_step_length = step_length
        self.ocp.solver_options.alpha_min = alpha_min
        self.ocp.solver_options.alpha_reduction = alpha_reduction

        # SQP solver tolerance
        self.ocp.solver_options.nlp_solver_tol_stat = nlp_solver_tol_stat
        self.ocp.solver_options.nlp_solver_tol_eq = nlp_solver_tol_eq
        self.ocp.solver_options.nlp_solver_tol_ineq = nlp_solver_tol_ineq
        self.ocp.solver_options.nlp_solver_tol_comp = nlp_solver_tol_comp
        # QP solver tolerance
        self.ocp.solver_options.qp_solver_tol_stat = qp_solver_tol_stat
        self.ocp.solver_options.qp_solver_tol_eq = qp_solver_tol_eq
        self.ocp.solver_options.qp_solver_tol_ineq = qp_solver_tol_ineq
        self.ocp.solver_options.qp_solver_tol_comp = qp_solver_tol_comp

        integrator_type = "ERK"
        integrator_stages = 4
        integrator_steps = 1
        newton_iter = 3
        self.ocp.solver_options.tf = self.optimization_horizon
        self.ocp.solver_options.integrator_type = integrator_type
        self.ocp.solver_options.sim_method_num_stages = integrator_stages
        self.ocp.solver_options.sim_method_num_steps = integrator_steps
        self.ocp.solver_options.sim_method_newton_iter = newton_iter

        self.solver = AcadosOcpSolver(self.ocp)

    def calc_raceline(
            self,
            s: float,
            V: float,
            n: float,
            chi: float,
            ax: float,
            ay: float,
            safety_distance: float,
            prev_solution: dict,
            V_min: float = 5.0,
    ):
        V_max = self.vehicle_params['v_max']
        raceline = self.__gen_raceline(
            s=s,
            V=max(V, V_min),
            n=n,
            chi=chi,
            ax=ax,
            ay=ay,
            prev_solution=prev_solution if V > V_min else None,
            safety_distance=safety_distance,
            V_max=V_max,
        )

        # calculate cartesian coordinates
        raceline['s'] = raceline['s'] % self.track_handler.s[-1]
        raceline_cartesian = self.track_handler.sn2cartesian(raceline['s'], raceline['n'])
        raceline['x'] = raceline_cartesian[:, 0]
        raceline['y'] = raceline_cartesian[:, 1]
        raceline['z'] = raceline_cartesian[:, 2]

        # calculate approximate time vector of racing line
        rl_xyz = self.track_handler.sn2cartesian(raceline['s'], raceline['n'])
        rl_ds = np.sqrt(np.diff(rl_xyz[:, 0]) ** 2 + np.diff(rl_xyz[:, 1]) ** 2)
        V_mean = (raceline['V'][:-1] + raceline['V'][1:]) / 2.0
        raceline['t'] = np.concatenate((np.array([0.0]), np.cumsum(rl_ds / V_mean)))

        return raceline

    def __gen_raceline(
            self,
            s: float,
            V: float,
            n: float,
            chi: float,
            ax: float,
            ay: float,
            prev_solution: dict,
            safety_distance: float,
            V_max: float,
    ):
        N = self.ocp.dims.N
        horizon = self.ocp.solver_options.tf

        if prev_solution:
            s_prev = np.unwrap(
                prev_solution['s'],
                discont=self.track_handler.s[-1] / 2.0,
                period=self.track_handler.s[-1]
            )
            if s < s_prev[0]:
                if s < self.track_handler.s[-1] / 2.0:
                    s += self.track_handler.s[-1]
                else:
                    s_prev -= self.track_handler.s[-1]

            s_array = np.linspace(s, s + horizon, N)

            # initial guess based on previous solution
            V_array = np.interp(s_array, s_prev, prev_solution['V'])
            ax_array = np.interp(s_array, s_prev, prev_solution['ax'])
            n_array = np.interp(s_array, s_prev, prev_solution['n'])
            chi_array = np.interp(s_array, s_prev, prev_solution['chi'])
            ay_array = np.interp(s_array, s_prev, prev_solution['ay'])
            jx_array = np.interp(s_array, s_prev, prev_solution['jx'])
            jy_array = np.interp(s_array, s_prev, prev_solution['jy'])

        else:
            s_array = np.linspace(s, s + horizon, N)
            V_array = V * np.ones_like(s_array)
            n_array = n * np.ones_like(s_array)
            chi_array = np.zeros_like(s_array)
            ax_array = np.zeros_like(s_array)
            ay_array = V ** 2 * np.interp(
                s_array % self.track_handler.s[-1], self.track_handler.s,
                self.track_handler.Omega_z,
                period=self.track_handler.s[-1]
            )
            jx_array = np.zeros_like(s_array)
            jy_array = np.zeros_like(s_array)

        w_tr_left = np.interp(
            s_array % self.track_handler.s[-1],
            self.track_handler.s,
            self.track_handler.w_tr_left,
            period=self.track_handler.s[-1]
        )
        w_tr_right = np.interp(
            s_array % self.track_handler.s[-1],
            self.track_handler.s,
            self.track_handler.w_tr_right,
            period=self.track_handler.s[-1]
        )
        veh_width = self.vehicle_params['total_width']

        V_exceed = max(0.0, V - V_max)

        # fix initial state
        x0 = np.array([s, V, n, chi, ax, ay])
        self.solver.set(0, 'x', x0)
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)

        # initial guess and constraints on shooting nodes
        for i in range(0, N):

            if i > 0:
                x_guess = np.array([s_array[i], V_array[i], n_array[i], chi_array[i], ax_array[i], ay_array[i]])
                self.solver.set(i, 'x', x_guess)

                # state constraints
                self.solver.constraints_set(
                    stage_=i,
                    field_='lbx',
                    value_=np.array([w_tr_right[i] + veh_width / 2.0 + (safety_distance), - np.pi / 2.0])
                )
                self.solver.constraints_set(
                    stage_=i,
                    field_='ubx',
                    value_=np.array([w_tr_left[i] - veh_width / 2.0 - (safety_distance), np.pi / 2.0])
                )

            u_guess = np.array([jx_array[i], jy_array[i], V_exceed])
            self.solver.set(i, 'u', u_guess)
            # input constraints
            self.solver.constraints_set(
                stage_=i,
                field_='lbu',
                value_=np.array([0.0])
            )
            self.solver.constraints_set(
                stage_=i,
                field_='ubu',
                value_=np.array([V_exceed])
            )

            # polytopic constraints
            self.solver.constraints_set(
                stage_=i,
                field_='lg',
                value_=np.array([0.0])
            )
            self.solver.constraints_set(
                stage_=i,
                field_='ug',
                value_=np.array([V_max])
            )

        # solve OCP
        self.solver.solve()

        # extract solution
        X = np.zeros((N, 6))
        U = np.zeros((N, 3))

        if self.gg_mode == 'polar':
            Sl = np.zeros((N, 2))
            Su = np.zeros((N, 2))
        elif self.gg_mode == 'diamond':
            Sl = np.zeros((N, 4))
            Su = np.zeros((N, 4))

        for i in range(N):
            X[i, :] = self.solver.get(i, "x")
            U[i, :] = self.solver.get(i, "u")
            if i > 0:
                Sl[i, :] = self.solver.get(i, "sl")
                Su[i, :] = self.solver.get(i, "su")

        raceline = {
            's': X[:, 0],
            'V': X[:, 1],
            'n': X[:, 2],
            'chi': X[:, 3],
            'ax': X[:, 4],
            'ay': X[:, 5],
            'jx': U[:, 0],
            'jy': U[:, 1],
            'epsilon_V': U[:, 2],
            'epsilon_n': np.maximum(Sl[:, 0], Su[:, 0]),
        }

        if self.gg_mode == 'polar':
            raceline['epsilon_rho'] = np.maximum(Sl[:, 1], Su[:, 1])

        elif self.gg_mode == 'diamond':
            raceline['epsilon_a_x'] = np.maximum(Sl[:, 1], Su[:, 1])
            raceline['epsilon_a_y'] = np.maximum(Sl[:, 2], Su[:, 2])
            raceline['epsilon_a_xy'] = np.maximum(Sl[:, 3], Su[:, 3])

        return raceline

# EOF
