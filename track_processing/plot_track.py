import os
import sys

# examples
track_name = 'IMS_3d_smoothed.csv'
# track_name = 'LVMS_3d_smoothed.csv'
# track_name = 'mount_panorama_3d_smoothed.csv'

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
in_data_path = os.path.join(data_path, 'smoothed_track_data')
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D

track_handler = Track3D(path=os.path.join(in_data_path, track_name))

track_handler.visualize(threeD=True)