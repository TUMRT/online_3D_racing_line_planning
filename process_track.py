import os
import sys
from track3D import Track3D

# choose what kind of processing is desired
# options are: - '3d_track_bounds_to_3d': generate 3d track data out of 3d track bounds
#              - '2d_to_3d': generate 3d track data out of 2d track data with banking information (assume z=0)
processing_method = '3d_track_bounds_to_3d'

# examples
track_name_raw = 'monza_3d.csv'
track_name_processed = 'monza_3d_smoothed.csv'
step_size = 2.0  # in meter
visualize = True
ignore_banking = False  # sets phi and mu to zero (rotation around x- and y-axis)

# Dict for cost function of track smoothing.
weights = {
    'w_c': 1e0,  # deviation to measurements centerline
    'w_l': 1e0,  # deviation to measurements left bound
    'w_r': 1e0,  # deviation to measurements right bound
    'w_theta': 1e5,  # smoothness theta
    'w_mu': 1e4,  # smoothness mu
    'w_phi': 5e3,  # smoothness phi
    'w_nl': 1e-2,  # smoothness left bound
    'w_nr': 1e-2  # smoothness right bound
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '', 'data')
raw_data_path = os.path.join(data_path, 'raw_track_data')
tmp_data_path = os.path.join(dir_path, 'tmp_track_data')
processed_data_path = os.path.join(dir_path, '', 'data', 'processed_tracks', )
os.makedirs(tmp_data_path, exist_ok=True)

track_handler = Track3D()

# process data
if processing_method == '3d_track_bounds_to_3d':
    track_handler.generate_3d_from_3d_track_bounds(
        path=os.path.join(raw_data_path, track_name_raw),
        out_path=os.path.join(tmp_data_path, track_name_raw.split('.')[0] + '_3d.csv'),
        ignore_banking=ignore_banking,
        visualize=visualize
    )
elif processing_method == '2d_to_3d':
    track_handler.generate_3d_from_2d_reference_line(
        path=os.path.join(raw_data_path, track_name_raw),
        out_path=os.path.join(tmp_data_path, track_name_raw.split('.')[0] + '_3d.csv'),
        ignore_banking=ignore_banking,
        visualize=visualize
    )
else:
    sys.exit(f"Processing Method '{processing_method}' not known.")


# smooth data
track_handler.smooth_track(
    out_path=os.path.join(processed_data_path, track_name_processed),
    weights=weights,
    step_size=step_size,
    visualize=visualize
)
