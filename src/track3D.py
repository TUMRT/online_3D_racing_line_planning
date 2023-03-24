import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import casadi as ca
from scipy.spatial.transform import Rotation
import time
from typing import Optional, Union
import os.path

g_earth = 9.81


def rad2deg(rad):
    return rad*180/np.pi


def deg2rad(deg):
    return deg*180/np.pi


class Track3D:

    def __init__(self, path=None):
        self.__path = path

        if self.__path:
            self.__track_data_frame = pd.read_csv(path, sep=',')
            self.track_locked = True
        else:
            self.__track_data_frame = None
            self.track_locked = False

    @property
    def track_locked(self):
        return self.__track_locked

    @track_locked.setter
    def track_locked(self, value):
        if value: self.lock_track_data()
        self.__track_locked = value

    def lock_track_data(self):
        # discretized points of track spine
        self.s = self.__track_data_frame['s_m'].to_numpy()# discretization step size
        self.ds = np.mean(np.diff(self.s))
        self.x = self.__track_data_frame['x_m'].to_numpy()
        self.y = self.__track_data_frame['y_m'].to_numpy()
        self.z = self.__track_data_frame['z_m'].to_numpy()
        self.theta = self.__track_data_frame['theta_rad'].to_numpy()
        self.mu = self.__track_data_frame['mu_rad'].to_numpy()
        self.phi = self.__track_data_frame['phi_rad'].to_numpy()
        self.w_tr_right = self.__track_data_frame['w_tr_right_m'].to_numpy()
        self.w_tr_left = self.__track_data_frame['w_tr_left_m'].to_numpy()
        self.Omega_x = self.__track_data_frame['omega_x_radpm'].to_numpy()
        self.Omega_y = self.__track_data_frame['omega_y_radpm'].to_numpy()
        self.Omega_z = self.__track_data_frame['omega_z_radpm'].to_numpy()

        # derivatives of omega with finite differencing
        self.dOmega_x = np.diff(self.Omega_x) / self.ds
        self.dOmega_x = np.append(self.dOmega_x, self.dOmega_x[0])
        self.dOmega_y = np.diff(self.Omega_y) / self.ds
        self.dOmega_y = np.append(self.dOmega_y, self.dOmega_y[0])
        self.dOmega_z = np.diff(self.Omega_z) / self.ds
        self.dOmega_z = np.append(self.dOmega_z, self.dOmega_z[0])

        # track spine interpolator
        def concatenate_arr(arr):
            return np.concatenate((arr, arr[1:], arr[1:]))  # 2 track lengths
        s_augmented = np.concatenate((self.s, self.s[-1] + self.s[1:], 2*self.s[-1] + self.s[1:]))  # 2 track lengths

        # casadi interpolator instances
        self.x_interpolator = ca.interpolant('x', 'linear', [s_augmented], concatenate_arr(self.x))
        self.y_interpolator = ca.interpolant('y', 'linear', [s_augmented], concatenate_arr(self.y))
        self.z_interpolator = ca.interpolant('z', 'linear', [s_augmented], concatenate_arr(self.z))
        self.theta_interpolator = ca.interpolant('theta', 'linear', [s_augmented], concatenate_arr(self.theta))
        self.mu_interpolator = ca.interpolant('mu', 'linear', [s_augmented], concatenate_arr(self.mu))
        self.phi_interpolator = ca.interpolant('phi', 'linear', [s_augmented], concatenate_arr(self.phi))
        self.w_tr_right_interpolator = ca.interpolant('w_tr_right', 'linear', [s_augmented], concatenate_arr(self.w_tr_right))
        self.w_tr_left_interpolator = ca.interpolant('w_tr_left', 'linear', [s_augmented], concatenate_arr(self.w_tr_left))
        self.Omega_x_interpolator = ca.interpolant('omega_x', 'linear', [s_augmented], concatenate_arr(self.Omega_x))
        self.Omega_y_interpolator = ca.interpolant('omega_y', 'linear', [s_augmented], concatenate_arr(self.Omega_y))
        self.Omega_z_interpolator = ca.interpolant('omega_z', 'linear', [s_augmented], concatenate_arr(self.Omega_z))
        self.dOmega_x_interpolator = ca.interpolant('domega_x', 'linear', [s_augmented], concatenate_arr(self.dOmega_x))
        self.dOmega_y_interpolator = ca.interpolant('domega_y', 'linear', [s_augmented], concatenate_arr(self.dOmega_y))
        self.dOmega_z_interpolator = ca.interpolant('domega_z', 'linear', [s_augmented], concatenate_arr(self.dOmega_z))

    def generate_3d_from_2d_reference_line(self, path, out_path, ignore_banking=False, visualize=False):
        if self.track_locked:
            raise RuntimeError('The track is locked and cannot be changed.')
        print('Generating 3d track file from 2d reference line with banking information ...')
        raw_data_frame = pd.read_csv(path, sep=',')
        self.__path = path

        # Global coordinates.
        x_m = raw_data_frame['x_m'].to_numpy()
        y_m = raw_data_frame['y_m'].to_numpy()

        # Check if reference line is closed (last point equals first point)
        diff_p = np.sqrt((x_m[0] - x_m[-1]) ** 2 + (y_m[0] - y_m[-1]) ** 2)
        if diff_p < 1e-3:
            # trim last point if closed since the following calculations a non-closed race track
            trim = -1
            x_m = x_m[:trim]
            y_m = y_m[:trim]
        else:
            trim = None

        z_m = np.zeros(x_m.shape)
        point_m = np.vstack((x_m, y_m, z_m)).transpose()
        # Append one entry to ensure last point is the same as first.
        point_m = np.vstack((point_m, point_m[0]))

        # Generate s coordinate.
        s_m = np.zeros(1)
        s = 0
        for i in range(1, point_m.shape[0]):
            dpoint = point_m[i] - point_m[i - 1]
            s += np.linalg.norm(dpoint)
            s_m = np.concatenate((s_m, [s]))

        # Get Euler angles.
        # banking angle equals camber angle since slope is assumed to be zero
        try:
            phi_rad = raw_data_frame['banking_rad'].to_numpy()[:trim]
        except:
            phi_rad = np.zeros_like(x_m)
        if ignore_banking:
            phi_rad[:] = 0
        # reference line is assumed to lie in z=0 and slope is zero
        mu_rad = np.zeros(phi_rad.shape)
        theta_rad = np.zeros(phi_rad.shape)
        for i in range(0, len(point_m)-1):
            dpoint = point_m[i+1] - point_m[i]
            theta_rad[i] = np.arctan2(dpoint[1], dpoint[0])
        euler_angles_rad = np.vstack((theta_rad, mu_rad, phi_rad)).transpose()
        # Append one entry to ensure last point is the same as first.
        euler_angles_rad = np.vstack((euler_angles_rad, euler_angles_rad[0]))

        # Get derivatives of euler angles.
        ds = np.diff(s_m)
        deuler_angles_radpm = np.diff(np.unwrap(euler_angles_rad, axis=0), axis=0) / np.array([ds, ds, ds]).T
        deuler_angles_radpm = np.vstack((deuler_angles_radpm, deuler_angles_radpm[0]))

        # Get track width.
        w_tr_right_m = - np.abs(raw_data_frame['w_tr_right_m'].to_numpy()[:trim])  # right is negative
        w_tr_right_m = np.concatenate((w_tr_right_m, [w_tr_right_m[0]]))
        w_tr_left_m = np.abs(raw_data_frame['w_tr_left_m'].to_numpy()[:trim])  # left is positive
        w_tr_left_m = np.concatenate((w_tr_left_m, [w_tr_left_m[0]]))
        # Banking influence.
        w_tr_right_m /= np.cos(euler_angles_rad[:, 2])
        w_tr_left_m /= np.cos(euler_angles_rad[:, 2])

        # Calculate angular velocities of track coordinate frame.
        # Empty matrix for omega values.
        omega_radpm = np.zeros((euler_angles_rad[:, 0].shape[0], 3))
        # Calculation of omega values
        for i in range(0, euler_angles_rad[:, 0].shape[0]):
            Jac = self.get_jacobian_J(euler_angles_rad[:, 1][i], euler_angles_rad[:, 2][i])
            deuler = np.array([deuler_angles_radpm[:, 2][i], deuler_angles_radpm[:, 1][i], deuler_angles_radpm[:, 0][i]])
            omega_radpm[i] = Jac.dot(deuler)

        # Visualize.
        if visualize:
            fig = plt.figure('Angular information')
            ax = fig.add_subplot(311)
            ax.plot(s_m, euler_angles_rad[:, 0], label=r'$\theta$')
            ax.plot(s_m, euler_angles_rad[:, 1], label=r'$\mu$')
            ax.plot(s_m, euler_angles_rad[:, 2], label=r'$\phi$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(312)
            ax.plot(s_m, deuler_angles_radpm[:, 0], label=r'$\frac{d\theta}{ds}$')
            ax.plot(s_m, deuler_angles_radpm[:, 1], label=r'$\frac{d\mu}{ds}$')
            ax.plot(s_m, deuler_angles_radpm[:, 2], label=r'$\frac{d\phi}{ds}$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(313)
            ax.plot(s_m, omega_radpm[:, 0], label=r'$\Omega_x$')
            ax.plot(s_m, omega_radpm[:, 1], label=r'$\Omega_y$')
            ax.plot(s_m, omega_radpm[:, 2], label=r'$\Omega_z$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()
            plt.show()

        # Create new data frame.
        self.__track_data_frame = pd.DataFrame()
        self.__track_data_frame['s_m'] = s_m
        self.__track_data_frame['x_m'] = point_m[:, 0]
        self.__track_data_frame['y_m'] = point_m[:, 1]
        self.__track_data_frame['z_m'] = point_m[:, 2]
        self.__track_data_frame['theta_rad'] = euler_angles_rad[:, 0]
        self.__track_data_frame['mu_rad'] = euler_angles_rad[:, 1]
        self.__track_data_frame['phi_rad'] = euler_angles_rad[:, 2]
        self.__track_data_frame['dtheta_radpm'] = deuler_angles_radpm[:, 0]
        self.__track_data_frame['dmu_radpm'] = deuler_angles_radpm[:, 1]
        self.__track_data_frame['dphi_radpm'] = deuler_angles_radpm[:, 2]
        self.__track_data_frame['w_tr_right_m'] = w_tr_right_m
        self.__track_data_frame['w_tr_left_m'] = w_tr_left_m
        self.__track_data_frame['omega_x_radpm'] = omega_radpm[:, 0]
        self.__track_data_frame['omega_y_radpm'] = omega_radpm[:, 1]
        self.__track_data_frame['omega_z_radpm'] = omega_radpm[:, 2]

        self.__path = out_path
        self.__track_data_frame.to_csv(path_or_buf=self.__path, sep=',', index=False, float_format='%.6f')
        return

    def generate_3d_from_3d_track_bounds(self, path, out_path, ignore_banking=False, visualize=False):
        if self.track_locked:
            raise RuntimeError('The track is locked and cannot be changed.')
        print('Generating 3d track file from 3d track bounds ...')
        self.__path = path

        # import track bounds
        track_bound_frame = pd.read_csv(path, sep=',')
        right_track_bounds_m = np.stack(
            (
                track_bound_frame['right_bound_x'].to_numpy(),
                track_bound_frame['right_bound_y'].to_numpy(),
                track_bound_frame['right_bound_z'].to_numpy()
            )
        ).T
        left_track_bounds_m = np.stack(
            (
                track_bound_frame['left_bound_x'].to_numpy(),
                track_bound_frame['left_bound_y'].to_numpy(),
                track_bound_frame['left_bound_z'].to_numpy()
            )
        ).T

        # calculate center line
        center_line_m = (right_track_bounds_m + left_track_bounds_m) / 2.0

        # calculate track widths
        w_tr_left_m = np.linalg.norm(left_track_bounds_m - center_line_m, axis=1)
        w_tr_right_m = - np.linalg.norm(right_track_bounds_m - center_line_m, axis=1)

        # calculate normal vector (points to left bound)
        normal_vector = (left_track_bounds_m - center_line_m) / w_tr_left_m[:, np.newaxis]

        # calculate tangential vectors
        tangential_vector_not_normalized = np.diff(center_line_m, axis=0)
        tangential_vector = tangential_vector_not_normalized / np.linalg.norm(tangential_vector_not_normalized, axis=1)[
                                                               :, np.newaxis]
        tangential_vector = np.append(tangential_vector, tangential_vector[0][np.newaxis], axis=0)

        # calculate orthogonal vectors to tangential and normal vectors
        orthogonal_vector_not_normalized = np.cross(tangential_vector, normal_vector, axis=1)
        orthogonal_vector = orthogonal_vector_not_normalized / np.linalg.norm(orthogonal_vector_not_normalized, axis=1)[
                                                               :, np.newaxis]

        # stack vectors to rotation matrices and calculate euler angles
        rotation_matrices = np.stack((tangential_vector, normal_vector, orthogonal_vector), axis=1)
        euler_angles_rad = - Rotation.from_matrix(rotation_matrices).as_euler('zyx')

        # set mu, phi and z to 0.0 if 2d track is assumed
        if ignore_banking:
            euler_angles_rad[:, 1:] = 0.0
            center_line_m[:, 2] = 0.0

        # calculate s-coordinate
        s_m = np.cumsum(np.sqrt(np.sum(np.square(np.diff(center_line_m, axis=0)), axis=1)))
        s_m = np.insert(s_m, 0, 0.0)

        # angular velocities with respect to s
        ds = np.diff(s_m)
        deuler_angles_radpm = np.diff(np.unwrap(euler_angles_rad, axis=0), axis=0) / np.array([ds, ds, ds]).T
        deuler_angles_radpm = np.vstack((deuler_angles_radpm, deuler_angles_radpm[0]))

        # Calculate angular velocities of track coordinate frame
        # Empty matrix for omega values
        omega_radpm = np.zeros((euler_angles_rad[:, 0].shape[0], 3))
        # Calculation of omega values
        for i in range(0, euler_angles_rad[:, 0].shape[0]):
            Jac = self.get_jacobian_J(euler_angles_rad[:, 1][i], euler_angles_rad[:, 2][i])
            deuler = np.array(
                [deuler_angles_radpm[:, 2][i], deuler_angles_radpm[:, 1][i], deuler_angles_radpm[:, 0][i]])
            omega_radpm[i] = Jac.dot(deuler)

        if visualize:
            fig = plt.figure('Angular information')
            ax = fig.add_subplot(311)
            ax.plot(s_m, euler_angles_rad[:, 0], label=r'$\theta$')
            ax.plot(s_m, euler_angles_rad[:, 1], label=r'$\mu$')
            ax.plot(s_m, euler_angles_rad[:, 2], label=r'$\phi$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(312)
            ax.plot(s_m, deuler_angles_radpm[:, 0], label=r'$\frac{d\theta}{ds}$')
            ax.plot(s_m, deuler_angles_radpm[:, 1], label=r'$\frac{d\mu}{ds}$')
            ax.plot(s_m, deuler_angles_radpm[:, 2], label=r'$\frac{d\phi}{ds}$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(313)
            ax.plot(s_m, omega_radpm[:, 0], label=r'$\Omega_x$')
            ax.plot(s_m, omega_radpm[:, 1], label=r'$\Omega_y$')
            ax.plot(s_m, omega_radpm[:, 2], label=r'$\Omega_z$')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()
            plt.show()

        # Create new data frame.
        self.__track_data_frame = pd.DataFrame()
        self.__track_data_frame['s_m'] = s_m
        self.__track_data_frame['x_m'] = center_line_m[:, 0]
        self.__track_data_frame['y_m'] = center_line_m[:, 1]
        self.__track_data_frame['z_m'] = center_line_m[:, 2]
        self.__track_data_frame['theta_rad'] = euler_angles_rad[:, 0]
        self.__track_data_frame['mu_rad'] = euler_angles_rad[:, 1]
        self.__track_data_frame['phi_rad'] = euler_angles_rad[:, 2]
        self.__track_data_frame['dtheta_radpm'] = deuler_angles_radpm[:, 0]
        self.__track_data_frame['dmu_radpm'] = deuler_angles_radpm[:, 1]
        self.__track_data_frame['dphi_radpm'] = deuler_angles_radpm[:, 2]
        self.__track_data_frame['w_tr_right_m'] = w_tr_right_m
        self.__track_data_frame['w_tr_left_m'] = w_tr_left_m
        self.__track_data_frame['omega_x_radpm'] = omega_radpm[:, 0]
        self.__track_data_frame['omega_y_radpm'] = omega_radpm[:, 1]
        self.__track_data_frame['omega_z_radpm'] = omega_radpm[:, 2]

        self.__path = out_path
        self.__track_data_frame.to_csv(path_or_buf=self.__path, sep=',', index=False, float_format='%.6f')

    def smooth_track(self, out_path: str, weights: dict, step_size=2.0, visualize=False, in_path=None):
        if self.track_locked:
            raise RuntimeError('The track is locked and cannot be changed.')
        print('Smoothing 3d track ...')
        if in_path is not None:
            self.__path = in_path
        # Load data frame.
        self.__track_data_frame = pd.read_csv(self.__path, sep=',')

        s_m_raw = self.__track_data_frame['s_m'].to_numpy()
        x_m_raw = self.__track_data_frame['x_m'].to_numpy()
        y_m_raw = self.__track_data_frame['y_m'].to_numpy()
        z_m_raw = self.__track_data_frame['z_m'].to_numpy()
        theta_rad_raw = np.unwrap(self.__track_data_frame['theta_rad'].to_numpy())
        mu_rad_raw = self.__track_data_frame['mu_rad'].to_numpy()
        phi_rad_raw = self.__track_data_frame['phi_rad'].to_numpy()
        dtheta_radpm_raw = self.__track_data_frame['dtheta_radpm'].to_numpy()
        dmu_radpm_raw = self.__track_data_frame['dmu_radpm'].to_numpy()
        dphi_radpm_raw = self.__track_data_frame['dphi_radpm'].to_numpy()
        w_tr_left_m_raw = self.__track_data_frame['w_tr_left_m'].to_numpy()
        w_tr_right_m_raw = self.__track_data_frame['w_tr_right_m'].to_numpy()
        omega_x_radpm_raw = self.__track_data_frame['omega_x_radpm'].to_numpy()
        omega_y_radpm_raw = self.__track_data_frame['omega_y_radpm'].to_numpy()
        omega_z_radpm_raw = self.__track_data_frame['omega_z_radpm'].to_numpy()
        
        # used s-coordinate
        s_m = np.linspace(s_m_raw[0], s_m_raw[-1], int(np.ceil(s_m_raw[-1] / step_size)) + 1)
        # actual step size
        step_size = np.mean(np.diff(s_m))
        
        # Check if reference line is closed (last point equals first point)
        diff_p = np.sqrt((x_m_raw[0] - x_m_raw[-1]) ** 2 + (y_m_raw[0] - y_m_raw[-1]) ** 2 + (z_m_raw[0] - z_m_raw[-1]) ** 2)
        if diff_p >= 1e-3: raise RuntimeError('Track must be closed for smoothing.')

        # State variables.
        # Scaling factors.
        px_s, py_s, pz_s, theta_s, mu_s, phi_s, dtheta_s, dmu_s, dphi_s, nl_s, nr_s = np.ones(11)
        # Euclidean coordinates.
        px_n = ca.MX.sym('px_n')
        px = px_s * px_n
        py_n = ca.MX.sym('py_n')
        py = py_s * py_n
        pz_n = ca.MX.sym('pz_n')
        pz = pz_s * pz_n
        # Euler angles.
        theta_n = ca.MX.sym('theta_n')
        theta = theta_s * theta_n
        mu_n = ca.MX.sym('mu_n')
        mu = mu_s * mu_n
        phi_n = ca.MX.sym('phi_n')
        phi = phi_s * phi_n
        # Euler angle derivatives.
        dtheta_n = ca.MX.sym('dtheta_n')
        dtheta = dtheta_s * dtheta_n
        dmu_n = ca.MX.sym('dmu_n')
        dmu = dmu_s * dmu_n
        dphi_n = ca.MX.sym('dphi_n')
        dphi = dphi_s * dphi_n
        # Track width.
        nl_n = ca.MX.sym('nl_n')
        nl = nl_s * nl_n
        nr_n = ca.MX.sym('nr_n')
        nr = nr_s * nr_n
        # State vector.
        x_s = np.array([px_s, py_s, pz_s, theta_s, mu_s, phi_s, dtheta_s, dmu_s, dphi_s, nl_s, nr_s])
        x = ca.vertcat(px_n, py_n, pz_n, theta_n, mu_n, phi_n, dtheta_n, dmu_n, dphi_n, nl_n, nr_n)
        nx = x.shape[0]

        # Control variables.
        # Scaling factors
        ddtheta_s, ddmu_s, ddphi_s, dnl_s, dnr_s = np.ones(5)
        # Euler anlge second derivatives.
        ddtheta_n = ca.MX.sym('ddtheta_n')
        ddtheta = ddtheta_s * ddtheta_n
        ddmu_n = ca.MX.sym('ddmu_n')
        ddmu = ddmu_s * ddmu_n
        ddphi_n = ca.MX.sym('ddphi_n')
        ddphi = ddphi_s * ddphi_n
        # Track width derivatives.
        dnl_n = ca.MX.sym('dnl_n')
        dnl = dnl_s * dnl_n
        dnr_n = ca.MX.sym('dnr_n')
        dnr = dnr_s * dnr_n
        # Control vector.
        u = ca.vertcat(ddtheta_n, ddmu_n, ddphi_n, dnl_n, dnr_n)
        nu = u.shape[0]

        # System dynamics.
        dx = ca.vertcat(ca.cos(theta)*ca.cos(mu), ca.sin(theta)*ca.cos(mu), -ca.sin(mu),
                        dtheta, dmu, dphi, ddtheta, ddmu, ddphi, dnl, dnr) / x_s

        # Get second derivatives of euler angles.
        ds_m_raw = np.diff(s_m_raw)
        ddtheta_radpm2_raw = np.diff(dtheta_radpm_raw) / ds_m_raw
        ddtheta_radpm2_raw = np.concatenate((ddtheta_radpm2_raw, [ddtheta_radpm2_raw[0]]))
        ddmu_radpm2_raw = np.diff(dmu_radpm_raw) / ds_m_raw
        ddmu_radpm2_raw = np.concatenate((ddmu_radpm2_raw, [ddmu_radpm2_raw[0]]))
        ddphi_radpm2_raw = np.diff(dphi_radpm_raw) / ds_m_raw
        ddphi_radpm2_raw = np.concatenate((ddphi_radpm2_raw, [ddphi_radpm2_raw[0]]))
        # Get derivatives of track width.
        dw_tr_left_m_raw = np.diff(w_tr_left_m_raw) / ds_m_raw
        dw_tr_left_m_raw = np.concatenate((dw_tr_left_m_raw, [dw_tr_left_m_raw[0]]))
        dw_tr_right_m_raw = np.diff(w_tr_right_m_raw) / ds_m_raw
        dw_tr_right_m_raw = np.concatenate((dw_tr_right_m_raw, [dw_tr_right_m_raw[0]]))
        # Interpolate functions.
        x_m_interp = ca.interpolant('x_m', 'linear', [s_m_raw], x_m_raw)
        y_m_interp = ca.interpolant('y_m', 'linear', [s_m_raw], y_m_raw)
        z_m_interp = ca.interpolant('z_m', 'linear', [s_m_raw], z_m_raw)
        theta_rad_interp = ca.interpolant('theta_m', 'linear', [s_m_raw], theta_rad_raw)
        mu_rad_interp = ca.interpolant('mu_m', 'linear', [s_m_raw], mu_rad_raw)
        phi_rad_interp = ca.interpolant('phi_m', 'linear', [s_m_raw], phi_rad_raw)
        dtheta_radpm_interp = ca.interpolant('dtheta_m', 'linear', [s_m_raw], dtheta_radpm_raw)
        dmu_radpm_interp = ca.interpolant('dmu_m', 'linear', [s_m_raw], dmu_radpm_raw)
        dphi_radpm_interp = ca.interpolant('dphi_m', 'linear', [s_m_raw], dphi_radpm_raw)
        w_tr_left_m_interp = ca.interpolant('w_tr_left_m', 'linear', [s_m_raw], w_tr_left_m_raw)
        w_tr_right_m_interp = ca.interpolant('w_tr_right_m', 'linear', [s_m_raw], w_tr_right_m_raw)
        ddtheta_radpm2_interp = ca.interpolant('ddtheta_m', 'linear', [s_m_raw], ddtheta_radpm2_raw)
        ddmu_radpm2_interp = ca.interpolant('ddmu_m', 'linear', [s_m_raw], ddmu_radpm2_raw)
        ddphi_radpm2_interp = ca.interpolant('ddphi_m', 'linear', [s_m_raw], ddphi_radpm2_raw)
        dw_tr_left_m_interp = ca.interpolant('dw_tr_left_m', 'linear', [s_m_raw], dw_tr_left_m_raw)
        dw_tr_right_m_interp = ca.interpolant('dw_tr_right_m', 'linear', [s_m_raw], dw_tr_right_m_raw)
        omega_x_radpm_interp = ca.interpolant('omega_x_m', 'linear', [s_m_raw], omega_x_radpm_raw)
        omega_y_radpm_interp = ca.interpolant('omega_y_m', 'linear', [s_m_raw], omega_y_radpm_raw)
        omega_z_radpm_interp = ca.interpolant('omega_z_m', 'linear', [s_m_raw], omega_z_radpm_raw)

        # Objective function.
        s = ca.MX.sym('s')
        # smoothed boundaries
        bl = ca.vertcat(px, py, pz) + self.get_normal_vector_casadi(theta, mu, phi) * nl
        br = ca.vertcat(px, py, pz) + self.get_normal_vector_casadi(theta, mu, phi) * nr
        # measured boundaries
        bl_bar = ca.vertcat(x_m_interp(s), y_m_interp(s), z_m_interp(s)) + self.get_normal_vector_casadi(theta_rad_interp(s), mu_rad_interp(s), phi_rad_interp(s)) * w_tr_left_m_interp(s)
        br_bar = ca.vertcat(x_m_interp(s), y_m_interp(s), z_m_interp(s)) + self.get_normal_vector_casadi(theta_rad_interp(s), mu_rad_interp(s), phi_rad_interp(s)) * w_tr_right_m_interp(s)

        L_e = weights['w_c'] * ((px - x_m_interp(s))**2 + (py - y_m_interp(s))**2 + (pz - z_m_interp(s))**2) + \
              weights['w_l'] * ca.dot(bl-bl_bar, bl-bl_bar) + weights['w_r'] * ca.dot(br-br_bar, br-br_bar)
        L_rc = weights['w_theta'] * ddtheta**2 + weights['w_mu'] * ddmu**2 + weights['w_phi'] * ddphi**2
        L_rw = weights['w_nl'] * dnl**2 + weights['w_nr'] * dnr**2
        L = L_e + L_rc + L_rw

        # Discrete time dynamics using fixed step Runge-Kutta 4 integrator.
        M = 4  # RK4 steps per interval
        ds_rk = step_size / M  # Step size used for RK integration
        f = ca.Function('f', [x, u, s], [dx, L])  # Function returning derivative and cost function
        X0 = ca.MX.sym('X0', nx)  # Input state.
        U = ca.MX.sym('U', nu)  # Input control.
        S0 = ca.MX.sym('S0')  # Input longitudinal position.
        S = S0
        X = X0  # Integrated derivative.
        Q = 0  # Integrated cost function.
        for j in range(M):
            k1, k1_q = f(X, U, S)
            k2, k2_q = f(X + ds_rk / 2 * k1, U, S + ds_rk / 2)
            k3, k3_q = f(X + ds_rk / 2 * k2, U, S + ds_rk / 2)
            k4, k4_q = f(X + ds_rk * k3, U, S + ds_rk)
            X = X + ds_rk / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + ds_rk / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            S = S + ds_rk
        # Function to integrate derivative and costs.
        F = ca.Function('F', [X0, U, S0], [X, Q], ['x0', 'p', 's0'], ['xf', 'qf'])

        # Start with an empty NLP.
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', nx)
        w += [Xk]
        lbw += [-np.inf, -np.inf, -np.inf, -np.inf, -np.pi/2.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        ubw += [np.inf, np.inf, np.inf, np.inf, np.pi/2.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        w0 += [x_m_interp(0.0)/px_s, y_m_interp(0.0)/py_s, z_m_interp(0.0)/pz_s, theta_rad_interp(0.0)/theta_s,
               mu_rad_interp(0.0)/mu_s, phi_rad_interp(0.0)/phi_s, dtheta_radpm_interp(0.0)/dtheta_s,
               dmu_radpm_interp(0.0)/dmu_s, dphi_radpm_interp(0.0)/dphi_s, w_tr_left_m_interp(0.0)/nl_s,
               w_tr_right_m_interp(0.0)/nr_s]

        for k in range(s_m.size - 1):
            # New NLP variable for the control.
            Uk = ca.MX.sym('U_' + str(k), nu)
            w += [Uk]
            lbw += [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
            ubw += [np.inf, np.inf, np.inf, np.inf, np.inf]
            w0 += [ddtheta_radpm2_interp(k*step_size)/ddtheta_s, ddmu_radpm2_interp(k*step_size)/ddmu_s, 
                   ddphi_radpm2_interp(k*step_size)/ddphi_s, dw_tr_left_m_interp(k*step_size)/dnl_s, 
                   dw_tr_right_m_interp(k*step_size)/dnr_s]

            # Integrate till the end of the interval.
            Fk = F(x0=Xk, p=Uk, s0=k * step_size)
            Xk_end = Fk['xf']
            J = J + Fk['qf']

            # New NLP variable for state at end of interval.
            Xk = ca.MX.sym('X_' + str(k + 1), nx)
            w += [Xk]
            lbw += [-np.inf, -np.inf, -np.inf, -np.inf, -np.pi/2.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
            ubw += [np.inf, np.inf, np.inf, np.inf, np.pi/2.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            w0 += [x_m_interp((k+1)*step_size)/px_s, y_m_interp((k+1)*step_size)/py_s, z_m_interp((k+1)*step_size)/pz_s,
                   theta_rad_interp((k+1)*step_size)/theta_s, mu_rad_interp((k+1)*step_size)/mu_s,
                   phi_rad_interp((k+1)*step_size)/phi_s, dtheta_radpm_interp((k+1)*step_size)/dtheta_s,
                   dmu_radpm_interp((k+1)*step_size)/dmu_s, dphi_radpm_interp((k+1)*step_size)/dphi_s,
                   w_tr_left_m_interp((k+1)*step_size)/nl_s, w_tr_right_m_interp((k+1)*step_size)/nr_s]

            # Add equality constraint for continuity.
            g += [Xk_end - Xk]
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # Boundary constraint: start states = final states.
        g += [w[0] - Xk]
        lbg += [0.0, 0.0, 0.0, -2.0*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0, 2.0*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Concatenate NLP vectors.
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w0 = ca.vertcat(*w0)
        lbw = ca.vertcat(lbw)
        ubw = ca.vertcat(ubw)
        lbg = ca.vertcat(lbg)
        ubg = ca.vertcat(ubg)

        # Create an NLP solver.
        nlp = {'f': J, 'x': w, 'g': g}
        opts = {"expand": False,
                "verbose": False,
                }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)  # ipopt

        # Solve the NLP.
        print('Start solving the NLP.')
        t_start = time.time()
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        t_end = time.time()
        print('Finished solving the NLP in {:5.3f}s.'.format(t_end - t_start))

        # Obtain solution.
        s_opt = s_m
        w_opt = sol['x'].full().flatten()
        x_opt = w_opt[0::(nx+nu)] * px_s
        y_opt = w_opt[1::(nx+nu)] * py_s
        z_opt = w_opt[2::(nx+nu)] * pz_s
        theta_opt = w_opt[3::(nx+nu)] * theta_s
        mu_opt = w_opt[4::(nx+nu)] * mu_s
        phi_opt = w_opt[5::(nx+nu)] * phi_s
        dtheta_opt = w_opt[6::(nx+nu)] * dtheta_s
        dmu_opt = w_opt[7::(nx+nu)] * dmu_s
        dphi_opt = w_opt[8::(nx+nu)] * dphi_s
        nl_opt = w_opt[9::(nx+nu)] * nl_s
        nr_opt = w_opt[10::(nx+nu)] * nr_s
        ddtheta_opt = w_opt[11::(nx+nu)] * ddtheta_s
        ddmu_opt = w_opt[12::(nx+nu)] * ddmu_s
        ddphi_opt = w_opt[13::(nx+nu)] * ddphi_s
        dnl_opt = w_opt[14::(nx+nu)] * dnl_s
        dnr_opt = w_opt[15::(nx+nu)] * dnr_s

        # Calculation of omega.
        # Empty matrix for omega values.
        omega_opt = np.zeros((s_opt.shape[0], 3))
        # Calculation of omega values
        for i in range(0, s_opt.shape[0]):
            Jac = self.get_jacobian_J(mu_opt[i], phi_opt[i])
            deuler = np.array([dphi_opt[i], dmu_opt[i], dtheta_opt[i]])
            omega_opt[i] = Jac.dot(deuler)

        # Plot track comparison.
        # Integrate theta.
        theta_integrated = np.zeros(theta_opt.shape[0])
        theta_integrated[0] = theta_opt[0]
        for i in range (1, theta_integrated.shape[0]):
            theta_integrated[i] = theta_integrated[i-1] + dtheta_opt[i-1]

        # Create new data frame.
        self.__track_data_frame = pd.DataFrame()
        self.__track_data_frame['s_m'] = s_opt
        self.__track_data_frame['x_m'] = x_opt
        self.__track_data_frame['y_m'] = y_opt
        self.__track_data_frame['z_m'] = z_opt
        self.__track_data_frame['theta_rad'] = theta_opt
        self.__track_data_frame['mu_rad'] = mu_opt
        self.__track_data_frame['phi_rad'] = phi_opt
        self.__track_data_frame['dtheta_radpm'] = dtheta_opt
        self.__track_data_frame['dmu_radpm'] = dmu_opt
        self.__track_data_frame['dphi_radpm'] = dphi_opt
        self.__track_data_frame['w_tr_right_m'] = nr_opt
        self.__track_data_frame['w_tr_left_m'] = nl_opt
        self.__track_data_frame['omega_x_radpm'] = omega_opt[:, 0]
        self.__track_data_frame['omega_y_radpm'] = omega_opt[:, 1]
        self.__track_data_frame['omega_z_radpm'] = omega_opt[:, 2]

        # Write data frame.
        self.__path = out_path
        self.__track_data_frame.to_csv(path_or_buf=self.__path, sep=',', index=False, float_format='%.6f')
        self.track_locked = True

        if visualize:
            # Angular information.
            fig = plt.figure("Angular information")
            ax = fig.add_subplot(311)
            ax.plot(s_opt, theta_opt, label=r'$\theta$', color='tab:blue')
            ax.plot(s_opt, theta_rad_interp(s_opt).full().flatten(), color='tab:blue', alpha=0.3)
            ax.plot(s_opt, mu_opt, label=r'$\mu$', color='tab:orange')
            ax.plot(s_opt, mu_rad_interp(s_opt).full().flatten(), color='tab:orange', alpha=0.3)
            ax.plot(s_opt, phi_opt, label=r'$\phi$', color='tab:green')
            ax.plot(s_opt, phi_rad_interp(s_opt).full().flatten(), color='tab:green', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(312)
            ax.plot(s_opt, dtheta_opt, label=r'$\frac{d\theta}{ds}$', color='tab:blue')
            ax.plot(s_opt, dtheta_radpm_interp(s_opt).full().flatten(), color='tab:blue', alpha=0.3)
            ax.plot(s_opt, dmu_opt, label=r'$\frac{d\mu}{ds}$', color='tab:orange')
            ax.plot(s_opt, dmu_radpm_interp(s_opt).full().flatten(), color='tab:orange', alpha=0.3)
            ax.plot(s_opt, dphi_opt, label=r'$\frac{d\phi}{ds}$', color='tab:green')
            ax.plot(s_opt, dphi_radpm_interp(s_opt).full().flatten(), color='tab:green', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(313)
            ax.plot(s_opt, omega_opt[:, 0], label=r'$\Omega_x$', color='tab:green')
            ax.plot(s_opt, omega_x_radpm_interp(s_opt).full().flatten(), color='tab:green', alpha=0.3)
            ax.plot(s_opt, omega_opt[:, 1], label=r'$\Omega_y$', color='tab:orange')
            ax.plot(s_opt, omega_y_radpm_interp(s_opt).full().flatten(), color='tab:orange', alpha=0.3)
            ax.plot(s_opt, omega_opt[:, 2], label=r'$\Omega_z$', color='tab:blue')
            ax.plot(s_opt, omega_z_radpm_interp(s_opt).full().flatten(), color='tab:blue', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()

            # Error of road boundaries.
            p_opt = np.array([x_opt, y_opt, z_opt])
            opt_normal = self.get_normal_vector_numpy(theta=theta_opt, mu=mu_opt, phi=phi_opt)
            left_opt = p_opt + opt_normal * nl_opt
            right_opt = p_opt + opt_normal * nr_opt
            p_old = np.array([x_m_interp(s_opt).full().flatten(),
                              y_m_interp(s_opt).full().flatten(),
                              z_m_interp(s_opt).full().flatten()])
            theta_old = theta_rad_interp(s_opt).full().flatten()
            mu_old = mu_rad_interp(s_opt).full().flatten()
            phi_old = phi_rad_interp(s_opt).full().flatten()
            old_normal = self.get_normal_vector_numpy(theta=theta_old, mu=mu_old, phi=phi_old)
            left_old = p_old + opt_normal * w_tr_left_m_interp(s_opt).full().flatten()
            right_old = p_old + old_normal * w_tr_right_m_interp(s_opt).full().flatten()
            fig = plt.figure("Boundary Errors")
            ax = fig.add_subplot(411)
            ax.plot(s_opt, np.linalg.norm(right_opt-right_old, axis=0), label='Right boundary', color='tab:red')
            ax.plot(s_opt, np.linalg.norm(left_opt-left_old, axis=0), label='Left boundary', color='tab:cyan')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(412)
            ax.plot(s_opt, right_opt[0] - right_old[0], label='Right boundary x', color='tab:red')
            ax.plot(s_opt, left_opt[0] - left_old[0], label='Left boundary x', color='tab:cyan')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(413)
            ax.plot(s_opt, right_opt[1] - right_old[1], label='Right boundary y', color='tab:red')
            ax.plot(s_opt, left_opt[1] - left_old[1], label='Left boundary y', color='tab:cyan')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(414)
            ax.plot(s_opt, right_opt[2] - right_old[2], label='Right boundary z', color='tab:red')
            ax.plot(s_opt, left_opt[2] - left_old[2], label='Left boundary z', color='tab:cyan')
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()

            # Track width.
            fig = plt.figure("Track width")
            ax = fig.add_subplot(211)
            ax.plot(s_opt, nr_opt, label='Width right', color='tab:red')
            ax.plot(s_opt, w_tr_right_m_interp(s_opt).full().flatten(), color='tab:red', alpha=0.3)
            ax.plot(s_opt, nl_opt, label='Width left', color='tab:cyan')
            ax.plot(s_opt, w_tr_left_m_interp(s_opt).full().flatten(), color='tab:cyan', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax = fig.add_subplot(212)
            ax.plot(s_opt, np.concatenate([dnr_opt, [0]]), label='d/ds Width right', color='tab:red')
            ax.plot(s_opt, dw_tr_right_m_interp(s_opt).full().flatten(), color='tab:red', alpha=0.3)
            ax.plot(s_opt, np.concatenate([dnl_opt, [0]]), label='d/ds Width left', color='tab:cyan')
            ax.plot(s_opt, dw_tr_left_m_interp(s_opt).full().flatten(), color='tab:cyan', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()

            plt.show()

    def visualize(self, show: bool = True, ax=None):
        if not self.track_locked:
            raise RuntimeError('Cannot visualize. Track is not locked.')

        # left and right bound
        left, right = self.get_track_bounds()

        # Create figure.
        if ax is None:
            fig = plt.figure('Track')
            ax_track = plt.axes(projection='3d')
        else:
            ax_track = ax

        ax_track.grid()
        ax_track.set_box_aspect((np.ptp(left[0]), np.ptp(left[1]), np.ptp(left[2])))  # aspect ratio is 1:1:1 in data space


        # Plot left border.
        ax_track.plot(left[0], left[1], left[2], 'k')
        # Plot right border.
        ax_track.plot(right[0], right[1], right[2], 'k')

        fig, ax = plt.subplots(nrows=3, num='Track angular information')
        ax[0].grid()
        ax[0].plot(self.s, rad2deg(self.theta), label=r'$\theta$')
        ax[0].plot(self.s, rad2deg(self.mu), label=r'$\mu$')
        ax[0].plot(self.s, rad2deg(self.phi), label=r'$\phi$')
        ax[0].set_ylabel(r'deg')
        ax[0].legend()

        ax[1].grid()
        ax[1].plot(self.s, rad2deg(self.Omega_x), label=r'$\Omega_x$')
        ax[1].plot(self.s, rad2deg(self.Omega_y), label=r'$\Omega_y$')
        ax[1].plot(self.s, rad2deg(self.Omega_z), label=r'$\Omega_z$')
        ax[1].set_ylabel(r'$\frac{\mathrm{deg}}{\mathrm{m}}$')
        ax[1].legend()

        ax[2].grid()
        ax[2].plot(self.s, rad2deg(self.dOmega_x), label=r'$\frac{d \Omega_x}{ds}$')
        ax[2].plot(self.s, rad2deg(self.dOmega_y), label=r'$\frac{d \Omega_y}{ds}$')
        ax[2].plot(self.s, rad2deg(self.dOmega_z), label=r'$\frac{d \Omega_z}{ds}$')
        ax[2].set_ylabel(r'$\frac{\mathrm{deg}}{\mathrm{m}²}$')
        ax[2].set_xlabel(r'$s$ in m')
        ax[2].legend()

        if show: plt.show()
        return ax_track

    def sn2cartesian(self, s, n, normal_vector_factor: float = 1.0):
        if not self.track_locked:
            raise RuntimeError('Cannot transform. Track is not locked.')
        euler_p = np.array([
            self.theta_interpolator(s),
            self.mu_interpolator(s),
            self.phi_interpolator(s)
        ]).squeeze()
        ref_p = np.array([
            self.x_interpolator(s),
            self.y_interpolator(s),
            self.z_interpolator(s)
        ]).squeeze().transpose()

        return ref_p + (self.get_normal_vector_numpy(*euler_p) * normal_vector_factor * n).transpose()

    def calc_apparent_accelerations(
            self, V, n, chi, ax, ay, s, h,
            neglect_w_omega_y: bool = True, neglect_w_omega_x: bool = True, neglect_euler: bool = True,
            neglect_centrifugal: bool = True, neglect_w_dot: bool = False, neglect_V_omega: bool = False,
    ):
        if not self.track_locked:
            raise RuntimeError('Cannot calculate apparent accelerations. Track is not locked.')
        
        mu = self.mu_interpolator(s)
        phi = self.phi_interpolator(s)
        Omega_x = self.Omega_x_interpolator(s)
        dOmega_x = self.dOmega_x_interpolator(s)
        Omega_y = self.Omega_y_interpolator(s)
        dOmega_y = self.dOmega_y_interpolator(s)
        Omega_z = self.Omega_z_interpolator(s)
        dOmega_z = self.dOmega_z_interpolator(s)

        s_dot = (V * ca.cos(chi)) / (1.0 - n * Omega_z)
        w = n * Omega_x * s_dot

        V_dot = ax
        if not neglect_w_omega_y:
            V_dot += w * (Omega_x * ca.sin(chi) - Omega_y * ca.cos(chi)) * s_dot

        n_dot = V * ca.sin(chi)

        chi_dot = ay / V - Omega_z * s_dot
        if not neglect_w_omega_x:
            chi_dot += w * (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot / V

        s_ddot = ((V_dot * ca.cos(chi) - V * ca.sin(chi) * chi_dot) * (1.0 - n * Omega_z) - (V * ca.cos(chi)) * (- n_dot * Omega_z - n * dOmega_z * s_dot)) / (1.0 + 2.0 * n * Omega_z + n ** 2 * Omega_z ** 2)

        omega_x_dot = 0.0
        omega_y_dot = 0.0
        if not neglect_euler:
            omega_x_dot = (dOmega_x * s_dot * ca.cos(chi) - Omega_x * ca.sin(chi) * chi_dot + dOmega_y * s_dot * ca.sin(chi) + Omega_y * ca.cos(chi) * chi_dot) * s_dot + (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_ddot
            omega_y_dot = (-dOmega_x * s_dot * ca.sin(chi) - Omega_x * ca.cos(chi) * chi_dot + dOmega_y * s_dot * ca.cos(chi) - Omega_y * ca.sin(chi) * chi_dot) * s_dot + (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_ddot

        omega_x = 0.0
        omega_y = 0.0
        omega_z = 0.0
        if not neglect_centrifugal:
            omega_x = (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot
            omega_y = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot
            omega_z = Omega_z * s_dot + chi_dot

        w_dot = 0.0
        if not neglect_w_dot:
            w_dot = n_dot * Omega_x * s_dot + n * dOmega_x * s_dot ** 2 + n * Omega_x * s_ddot

        V_omega = 0.0
        if not neglect_V_omega:
            V_omega = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot * V

        ax_tilde = ax + omega_y_dot * h - omega_z * omega_x * h + g_earth * (- ca.sin(mu) * ca.cos(chi) + ca.cos(mu) * ca.sin(phi) * ca.sin(chi))
        ay_tilde = ay + omega_x_dot * h + omega_z * omega_y * h + g_earth * (ca.sin(mu) * ca.sin(chi) + ca.cos(mu) * ca.sin(phi) * ca.cos(chi))
        g_tilde = ca.fmax(w_dot - V_omega + (omega_x ** 2 - omega_y ** 2) * h + g_earth * ca.cos(mu) * ca.cos(phi), 0.0)

        return ax_tilde, ay_tilde, g_tilde
 
    def get_track_bounds(self, margin=0.0):
        normal_vector = self.get_normal_vector_numpy(self.theta, self.mu, self.phi)
        left = np.array([self.x + normal_vector[0] * (self.w_tr_left + margin),
                         self.y + normal_vector[1] * (self.w_tr_left + margin),
                         self.z + normal_vector[2] * (self.w_tr_left + margin)])
        right = np.array([self.x + normal_vector[0] * (self.w_tr_right - margin),
                          self.y + normal_vector[1] * (self.w_tr_right - margin),
                          self.z + normal_vector[2] * (self.w_tr_right - margin)])
        return left, right

    @staticmethod
    def get_rotation_matrix_numpy(theta, mu, phi):
        return np.array([
            [np.cos(theta) * np.cos(mu),  np.cos(theta) * np.sin(mu) * np.sin(phi) - np.sin(theta) * np.cos(phi), np.cos(theta) * np.sin(mu) * np.cos(phi) + np.sin(theta) * np.sin(phi)],
            [np.sin(theta) * np.cos(mu), np.sin(theta) * np.sin(mu) * np.sin(phi) + np.cos(theta) * np.cos(phi), np.sin(theta) * np.sin(mu) * np.cos(phi) - np.cos(theta) * np.sin(phi)],
            [- np.sin(mu), np.cos(mu) * np.sin(phi), np.cos(mu) * np.cos(phi)]
        ]).squeeze()

    @staticmethod
    def get_normal_vector_numpy(theta, mu, phi):
        return Track3D.get_rotation_matrix_numpy(theta, mu, phi)[:, 1]

    @staticmethod
    def get_normal_vector_casadi(theta, mu, phi):
        return ca.vertcat(
            ca.cos(theta) * ca.sin(mu) * ca.sin(phi) - ca.sin(theta) * ca.cos(phi),
            ca.sin(theta) * ca.sin(mu) * ca.sin(phi) + ca.cos(theta) * ca.cos(phi),
            ca.cos(mu) * ca.sin(phi)
        )

    @staticmethod
    def get_jacobian_J(mu, phi):
        return np.array([
            [1, 0, -np.sin(mu)],
            [0, np.cos(phi), np.cos(mu) * np.sin(phi)],
            [0, -np.sin(phi), np.cos(mu) * np.cos(phi)]
        ])

# EOF
