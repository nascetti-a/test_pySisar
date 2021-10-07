# This module contains a dictionary, cfg, containing all the parameters of the
# pySISAR pipeline. This dictionary is updated at runtime with parameters defined
# by the user in the config.json file. All the optional parameters (that the
# user is not forced to define in config.json) must be defined here, otherwise
# they won't have a default value.


cfg = {}

# path to output directory
cfg['out_dir'] = "./pyDate_output"

# path to directory where (many) temporary files will be stored
cfg['temporary_dir'] = "./pyDate_tmp"


# resolution of the output digital surface model, in meters per pixel
cfg['dsm_resolution'] = 2

# set the path to a geoid file in pgm format - default egm 2008 file available in the repository
cfg['geoid_path'] = '../geoid/egm2008-5.pgm'

# Select the dense matching method: 'SGM' or 'FLOW'
cfg['dense_matching_method'] = 'SGM'

# Select if 'OPTICAL' or 'SAR' sensor
cfg['sensor'] = 'OPTICAL'

# Select the sgm parameters
cfg['sgm_param'] = {'minDisp': -32, 'numDisp': 64, 'blockSize': 7, 'window_size': 3}

# Select the flow parameters
cfg['flow_param'] = {'window_size': 11, 'levels': 3}

# Select the
cfg['ncc_param'] = {'window_size': 11, "numDisp": 16, "threshold": 0.5}

