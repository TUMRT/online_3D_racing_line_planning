import os
import sys

# examples
track_name = 'LVMS_3d.csv'
track_name_processed = 'LVMS_3d_smoothed.csv'

# track_name = 'mount_panorama_3d.csv'
# track_name_processed = 'mount_panorama_3d_smoothed.csv'


# Dictionary for cost function of track smoothing.
weights = {
    'w_c': 1e0,  # deviation to measurements centerline
    'w_l': 1e0,  # deviation to measurements left bound
    'w_r': 1e0,  # deviation to measurements right bound
    'w_theta': 1e7,  # smoothness theta
    'w_mu': 1e5,  # smoothness mu
    'w_phi': 1e4,  # smoothness phi
    'w_nl': 1e-2,  # smoothness left bound
    'w_nr': 1e-2  # smoothness right bound
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
in_data_path = os.path.join(data_path, '3d_track_data')
out_data_path = os.path.join(data_path, 'smoothed_track_data')
os.makedirs(out_data_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D

track_handler = Track3D()

track_handler.smooth_track(
    out_path=os.path.join(out_data_path, track_name_processed),
    weights=weights,
    in_path=os.path.join(in_data_path, track_name),
    visualize=True
)

