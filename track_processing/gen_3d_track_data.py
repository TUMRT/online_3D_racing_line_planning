import os
import sys

# choose what kind of processing is desired
# options are:  - '2d_to_3d': generate 3d track data out of 2d track data with banking information (assume z=0)
#               - '3d_bounds_to_3d': generate 3d track data out of 3d track bounds

# examples
track_name_raw = 'IMS_2d_centerline_banking.csv'
processing_method = '2d_to_3d'
track_name_processed = 'IMS_3d.csv'

"""
track_name_raw = 'LVMS_2d_centerline_banking.csv'
processing_method = '2d_to_3d'
track_name_processed = 'LVMS_3d.csv'
"""

"""
track_name_raw = 'mount_panorama_bounds_3d.csv'
processing_method = '3d_track_bounds_to_3d'
track_name_processed = 'mount_panorama_3d.csv'
"""

step_size = 2.0  # in meter
visualize = True
ignore_banking = False  # sets phi and mu to zero (rotation around x- and y-axis)

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
raw_data_path = os.path.join(data_path, 'raw_track_data')
out_data_path = os.path.join(data_path, '3d_track_data')
os.makedirs(out_data_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D

track_handler = Track3D()

# process data
if processing_method == '2d_to_3d':
    track_handler.generate_3d_from_2d_reference_line(
        path=os.path.join(raw_data_path, track_name_raw),
        out_path=os.path.join(out_data_path, track_name_processed),
        ignore_banking=ignore_banking,
        visualize=visualize
    )
elif processing_method == '3d_track_bounds_to_3d':
    track_handler.generate_3d_from_3d_track_bounds(
        path=os.path.join(raw_data_path, track_name_raw),
        out_path=os.path.join(out_data_path, track_name_processed),
        ignore_banking=ignore_banking,
        visualize=visualize
    )

