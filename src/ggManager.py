import numpy as np
import os.path
import casadi as ca

g_earth = 9.81

class GGManager():
    def __init__(
            self,
            gg_path: str,
            gg_margin: float = 0.05,
    ):
        self.gg_margin = gg_margin

        self.__load_gggv_data(gg_path)

        self.rho_interpolator_no_margin = self.__get_rho_interpolator(margin=False)
        self.rho_interpolator = self.__get_rho_interpolator(margin=True)
        
        self.gg_exponent_interpolator, self.ax_max_interpolator, self.ax_min_interpolator, self.ay_max_interpolator, self.acc_interpolator = \
            self.__get_diamond_interpolators()

    def __load_gggv_data(self, gggv_path):
        self.V_list = np.load(os.path.join(gggv_path, 'v_list.npy'))
        # add velocity = 0 and assume the same limits as for the smallest available velocity
        self.V_list = np.insert(self.V_list, 0, 0.0)
        self.V_max = self.V_list.max()
        self.g_list = np.load(os.path.join(gggv_path, 'g_list.npy'))
        # add g = 0 and assume rho = 0
        self.g_list = np.insert(self.g_list, 0, 0.0)
        self.g_max = self.g_list.max()
        # polar coordinates
        self.alpha_list = np.load(os.path.join(gggv_path, 'alpha_list.npy'))
        self.rho_list = np.load(os.path.join(gggv_path, 'rho.npy'))
        self.rho_list = np.insert(self.rho_list, 0, self.rho_list[0], axis=0)  # for added velocity = 0
        self.rho_list = np.insert(self.rho_list, 0, 1e-3*np.ones_like(self.rho_list[0, 1]), axis=1)  # for added g = 0
        # diamond approximation
        self.gg_exponent_list = np.load(os.path.join(gggv_path, 'gg_exponent.npy'))
        self.gg_exponent_list = np.insert(self.gg_exponent_list, 0, self.gg_exponent_list[0], axis=0)  # for added velocity = 0
        self.gg_exponent_list = np.insert(self.gg_exponent_list, 0, self.gg_exponent_list[0, 0], axis=1)  # for added g = 0 (use the same exponent as for the smallest available g)
        self.ax_min_list = np.load(os.path.join(gggv_path, 'ax_min.npy'))
        self.ax_min_list = np.insert(self.ax_min_list, 0, self.ax_min_list[0], axis=0)  # for added velocity = 0
        self.ax_min_list = np.insert(self.ax_min_list, 0, 1e-3*np.ones_like(self.ax_min_list[0, 1]), axis=1)  # for added g = 0
        self.ax_max_list = np.load(os.path.join(gggv_path, 'ax_max.npy'))
        self.ax_max_list = np.insert(self.ax_max_list, 0, self.ax_max_list[0], axis=0)  # for added velocity = 0
        self.ax_max_list = np.insert(self.ax_max_list, 0, 1e-3*np.ones_like(self.ax_max_list[0, 1]), axis=1)  # for added g = 0
        self.ay_max_list = np.load(os.path.join(gggv_path, 'ay_max.npy'))
        self.ay_max_list = np.insert(self.ay_max_list, 0, self.ay_max_list[0], axis=0)  # for added velocity = 0
        self.ay_max_list = np.insert(self.ay_max_list, 0, 1e-3*np.ones_like(self.ay_max_list[0, 1]), axis=1)  # for added g = 0

    def __get_diamond_interpolators(self):
        gg_exponent_interpolator = ca.interpolant(
            'gg_exponent_interpolator', 'linear', [self.V_list, self.g_list], self.gg_exponent_list.ravel(order='F')
        )
        ax_max_interpolator = ca.interpolant(
            'ax_max_interpolator', 'linear', [self.V_list, self.g_list], self.ax_max_list.ravel(order='F') * (1.0 - self.gg_margin)
        )
        ax_min_interpolator = ca.interpolant(
            'ax_min_interpolator', 'linear', [self.V_list, self.g_list], self.ax_min_list.ravel(order='F') * (1.0 - self.gg_margin)
        )
        ay_max_interpolator = ca.interpolant(
            'ay_max_interpolator', 'linear', [self.V_list, self.g_list], self.ay_max_list.ravel(order='F') * (1.0 - self.gg_margin)
        )

        acc_interpolator = ca.interpolant(
            'acc_interpolator', 'linear', [self.V_list, self.g_list],
            np.array([self.gg_exponent_list, self.ax_min_list * (1.0 - self.gg_margin), self.ax_max_list * (1.0 - self.gg_margin), self.ay_max_list * (1.0 - self.gg_margin)]).ravel(order='F')
        )
        return gg_exponent_interpolator, ax_max_interpolator, ax_min_interpolator, ay_max_interpolator, acc_interpolator

    def __get_rho_interpolator(self, margin: bool):
        # create interpolator
        factor = (1.0 - self.gg_margin) if margin else 1.0
        rho_interpolator = ca.interpolant(
            'rho_interpolator',
            'linear',
            [self.V_list, self.g_list, self.alpha_list],
            self.rho_list.ravel(order='F') * factor
        )
        return rho_interpolator