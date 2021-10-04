import cv2
import numpy as np
import json

from matplotlib import pyplot as plt
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pySISAR import align_and_compute_disp
from pySISAR import make_ortho
from pySISAR import coarse_dem
from pySISAR import save_raster_as_geotiff
from pySISAR import initialitaion
from pySISAR.config import cfg
from pySISAR import common


def main():

    """
        Launch the pyDATE pipeline with the parameters given in a json file.

        Args:
            config.json: user config dictionary
    """

    config_file = "config.json"

    # Initialiation of the input parameters
    with open(config_file, 'r') as f:
        user_cfg = json.load(f)
        initialitaion.initialize_config(user_cfg)

    print("pyDATE initialization complete")

    print("Loading input images...")

    img1 = cv2.imread(cfg['images'][0]['img'], cv2.IMREAD_ANYDEPTH)
    img2 = cv2.imread(cfg['images'][1]['img'], cv2.IMREAD_ANYDEPTH)

    print(type(img1), img1.shape)
    print(type(img2), img2.shape)

    rpc_img1 = cfg['images'][0]['rpcm']
    rpc_img2 = cfg['images'][1]['rpcm']

    grid = cfg['grid']

    # Retrieve the ortho grid for the DSM extraction
    print("Bounding Box: ")
    print(grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat)

    dem = coarse_dem(grid.ul_lon, grid.ul_lat, grid.delta_lon, grid.delta_lat)


    print("Mean Elevation of the area:")
    print(np.mean(dem[0]))
    #print(dem[0].shape)
    #print(dem[1])

    upsample_dem, dem_proj = common.scaling_geo_raster(dem[0], dem[1])

    #print(upsample_dem.shape)
    #print(dem_proj)

    #plt.subplot(121), plt.imshow(dem[0], vmin=np.percentile(dem[0], 2.5), vmax=np.percentile(dem[0], 97.5))
    #plt.subplot(122), plt.imshow(upsample_dem, vmin=np.percentile(upsample_dem, 2.5), vmax=np.percentile(upsample_dem, 97.5))
    #plt.show()

    print("Generating ortho image left...")
    ortho1 = make_ortho(grid.ul_lon, grid.lr_lon, grid.ul_lat, grid.lr_lat, grid.width, grid.height, grid.gsd, img1, rpc_img1.fast_rpc(), upsample_dem, dem_proj)
    print("Generating ortho image right..")
    ortho2 = make_ortho(grid.ul_lon, grid.lr_lon, grid.ul_lat, grid.lr_lat, grid.width, grid.height, grid.gsd, img2, rpc_img2.fast_rpc(), upsample_dem, dem_proj)

    save_raster_as_geotiff(ortho1[0], grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat, cfg['temporary_dir'] + "/approx_ortho_right.tiff")
    save_raster_as_geotiff(ortho2[0], grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat, cfg['temporary_dir'] + "/approx_ortho_left.tiff")

    eq_ortho1 = common.norm_img(ortho1[0])
    eq_ortho2 = common.norm_img(ortho2[0])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(ortho1[0], vmin=np.percentile(ortho1[0], 2.5), vmax=np.percentile(ortho1[0], 97.5))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(ortho2[0], vmin=np.percentile(ortho2[0], 2.5), vmax=np.percentile(ortho2[0], 97.5))
    plt.show()

    dense_results = align_and_compute_disp(eq_ortho1, eq_ortho2, rotate=cfg['rotate'])

    # This is an old implementaion of the conversion factor - it will be updated soon
    conv_factor = common.compute_conversion_factor(rpc_img2, rpc_img1, grid)

    if cfg['dense_matching_method'] == 'SGM':

        if cfg['sensor'] == 'OPTICAL':
            final_dem = ortho1[1] - (dense_results[0] / 16) / conv_factor[2]  # Check error in +/- changing orbit
            final_dem[np.where(dense_results[1] == 0)] = np.nan

        elif cfg['sensor'] == 'SAR':
            final_dem = ortho1[1] - ((dense_results[0] / 16) / conv_factor[0])  #*dense_results[1]/255  # Check error in +/- changing orbit
            #final_dem[np.where(dense_results[1] == 0)] = ortho1[1][np.where(dense_results[1] == 0)]
            #final_dem[np.where(dense_results[1] == 0)] = np.nan

        # TEST MGM
        # raster = gdal.Open("/Users/andreanascetti/PycharmProjects/s2p/3rdparty/mgm_multi/disp16.tif")
        # img_MGM = raster.GetRasterBand(1).ReadAsArray()
        # img_MGM = cv2.rotate(img_MGM, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # final_dem = ortho1[1] + (img_MGM) / conv_factor

    elif cfg['dense_matching_method'] == 'FLOW':
        final_dem = ortho1[1] - (dense_results[1])/conv_factor[2] + (dense_results[0])/conv_factor[1]

    elif cfg['dense_matching_method'] == 'NCC':
        final_dem = ortho1[1] + (dense_results[0]) / conv_factor[0]
        final_dem = cv2.blur(final_dem, (3, 3))

    save_raster_as_geotiff(final_dem, grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat, cfg['out_dir'] + "/finale_dem.tiff")

    plt.imshow(final_dem, vmin=np.nanpercentile(final_dem, 15.0), vmax=np.nanpercentile(final_dem, 95.0))
    plt.show()

    return 0


if __name__ == "__main__":
    main()
