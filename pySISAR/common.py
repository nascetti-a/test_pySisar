import os
import cv2
import numpy as np
import errno
import pygeodesy as geod

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class Grid:

    def __init__(self, roi, gsd):

        self.ul_lat = roi['UL_Lat']
        self.ul_lon = roi['UL_Lon']
        self.lr_lat = roi['LR_Lat']
        self.lr_lon = roi['LR_Lon']
        self.gsd = gsd

        UL_UTM = geod.toUtm8(roi['UL_Lat'], roi['UL_Lon'])
        LR_UTM = geod.toUtm8(roi['LR_Lat'], roi['LR_Lon'])

        self.width = int((LR_UTM.easting - UL_UTM.easting) / gsd)
        self.height = int((UL_UTM.northing - LR_UTM.northing) / gsd)

        self.delta_lat = self.ul_lat - self.lr_lat
        self.delta_lon = self.lr_lon - self.ul_lon


def norm_img(img, eq_min=2.5, eq_max=97.5):
    eq_range = [np.percentile(img, eq_min), np.percentile(img, eq_max)]
    return (img-eq_range[0])*(255/(eq_range[1]-eq_range[0]))

def norm_uint8(img):
    eq_range = [np.percentile(img, 2.5), np.percentile(img, 97.5)]
    eq_img = (img-eq_range[0])*(255/(eq_range[1]-eq_range[0]))
    eq_img[np.where(eq_img < 0)] = 0
    eq_img[np.where(eq_img > 255)] = 255
    return eq_img.astype("uint8")

def compute_conversion_factor(rpc_img1, rpc_img2, grid):

    grid_size = 3

    UL_I, UL_J = rpc_img1.projection(grid.ul_lon, grid.ul_lat, 200)  # CHECK ELEVATION
    LR_I, LR_J = rpc_img1.projection(grid.lr_lon, grid.lr_lat, 200)

    cols = np.linspace(UL_I, LR_I, grid_size)
    rows = np.linspace(LR_J, UL_J, grid_size)

    print(cols)
    print(rows)

    MinimumHeight = 220
    MaximumHeight = 320
    deltaH = MaximumHeight - MinimumHeight

    conversion_factors = []
    conversion_factors_East = []
    conversion_factors_Nord =[]

    for i in cols:
        for j in rows:

            groundPoint_master = rpc_img1.localization(i, j, MaximumHeight)
            groundPointDown = rpc_img1.localization(i, j, MinimumHeight)

            I_slave, J_slave = rpc_img2.projection(groundPointDown[0], groundPointDown[1], MinimumHeight)
            groundPoint_slave = rpc_img2.localization(I_slave, J_slave, MaximumHeight)

            UTMgroundPoint_master = geod.toUtm8(groundPoint_master[1], groundPoint_master[0])
            UTMgroundPoint_slave = geod.toUtm8(groundPoint_slave[1], groundPoint_slave[0])

            DE = UTMgroundPoint_slave.easting - UTMgroundPoint_master.easting
            DN = UTMgroundPoint_slave.northing - UTMgroundPoint_master.northing

            #print(DE, DN)
            #print(np.sqrt((DE * DE) + (DN * DN)) / deltaH)

            conversion_factors_East.append(DE/deltaH)
            conversion_factors_Nord.append(DN/deltaH)
            conversion_factors.append((np.sqrt((DE * DE) + (DN * DN))) / deltaH)

    print("Mean convertion factor: ", np.mean(conversion_factors))
    print("Mean convertion factor East: ", np.mean(conversion_factors_East))
    print("Mean convertion factor North: ", np.mean(conversion_factors_Nord))

    return np.mean(conversion_factors), np.mean(conversion_factors_East), np.mean(conversion_factors_Nord)


def scaling_sar(img, f_scale=2):

    d_size = (int(img.shape[1]/f_scale), int(img.shape[0]/f_scale))

    scale_img = cv2.resize(img, d_size, interpolation=cv2.INTER_AREA)

    scale_img = cv2.resize(scale_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return scale_img

def scaling_geo_raster(img, proj, f_scale=8):

    d_size = (img.shape[1]*f_scale, img.shape[0]*f_scale)

    scale_img = cv2.resize(img, d_size, interpolation=cv2.INTER_LANCZOS4)

    proj[1] = proj[1]/f_scale
    proj[5] = proj[5]/f_scale

    return scale_img, proj


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def enhance(image, clip_limit=3):
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image
