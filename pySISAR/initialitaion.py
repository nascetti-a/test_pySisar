import rpcm
import sys

from pySISAR import common
from pySISAR.config import cfg


def initialize_config(json_file):
    """
    Populate a dictionary containing the pyDate parameters from a user config file.
    This dictionary is contained in the global variable 'cfg' of the config
    module.

    Args:
        user_cfg: user config dictionary
    """

    check_parameters(json_file)

    cfg.update(json_file)

    cfg['grid'] = common.Grid(cfg['roi'], cfg['dsm_resolution'])

    make_dirs()

    if cfg['sensor'] == 'OPTICAL':
        cfg['rotate'] = True
        print("Across Track rotation is active")
    else:
        cfg['rotate'] = False
        print("Across Track rotation is not active")

    print_log()


def dict_has_keys(d, l):
    """
    Return True if the dict d contains all the keys of the input list l.
    """
    return all(k in d for k in l)


def check_parameters(d):
    """
    Check that the provided dictionary defines all mandatory pyDate arguments.
    Args:
        d: python dictionary
    """

    # verify that input files paths are defined
    if 'images' not in d or len(d['images']) < 2:
        print('ERROR: missing paths to input images')
        sys.exit(1)
    for img in d['images']:
        if not dict_has_keys(img, ['img']):
            print('ERROR: missing img paths for image', img)
            sys.exit(1)

    # read RPCs
    for img in d['images']:
        if 'rpc' in img:
            if isinstance(img['rpc'], str):  # path to an RPC file
                img['rpcm'] = rpcm.rpc_from_rpc_file(img['rpc'])
            elif isinstance(img['rpc'], dict):  # RPC dict in 'rpcm' format
                img['rpcm'] = rpcm.RPCModel(img['rpc'], dict_format='rpcm')
            else:
                raise NotImplementedError(
                    'rpc of type {} not supported'.format(type(img['rpc']))
                )
        else:
            img['rpcm'] = rpcm.rpc_from_geotiff(img['img'])

    if 'roi' not in d or len(d['roi']) != 4:
        print('ERROR: missing or wrong roi input')
        sys.exit(1)
    else:
        if 'UL_Lon' not in d['roi'] or 'UL_Lat' not in d['roi']:
            print('ERROR: Upper Right corner or wrong format')
            sys.exit(1)
        elif 'LR_Lon' not in d['roi'] or 'LR_Lat' not in d['roi']:
            print('ERROR: Lower Right corner or wrong format')
            sys.exit(1)
        else:
            print("Selected ROI:")
            print(d['roi'])

    if 'dense_matching_method' in d:
        if d['dense_matching_method'] != 'SGM' and d['dense_matching_method'] != 'FLOW' and d['dense_matching_method'] != 'NCC':
            print("Invalid matching method: please select one of the following sgm or flow")
            sys.exit(1)
        else:
            print("Selected matching method: ", d['dense_matching_method'])

    if 'sensor' in d:
        if d['sensor'] != 'SAR' and d['sensor'] != 'OPTICAL':
            print("Invalid sensor type: please select OPTICAL or SAR")
            sys.exit(1)
        else:
            print("Sensor type: ", d['sensor'])

    if 'sgm_option' in d:
        cfg['sgm_param'].update(d['sgm_option'])

    if 'flow_option' in d:
        cfg['flow_param'].update(d['flow_option'])


def make_dirs():
    """

    Make output and temporary directories to run pyDate

    """

    common.mkdir_p(cfg['out_dir'])
    common.mkdir_p(cfg['temporary_dir'])


def print_log():
    """

    Print log file with all the input parameters to run pyDate

    """

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    with open(cfg['out_dir']+'/log.txt', 'wt') as out:
        pprint.pprint(cfg, stream=out)

