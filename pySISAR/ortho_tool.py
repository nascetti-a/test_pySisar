import numpy as np
import math
from sardem.dem import main as load_dem
from numba import jit, prange
from typing import Tuple, NamedTuple
import gdal
import osr
import pygeodesy as geo
from pySISAR.config import cfg

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#print(cfg['temporary_dir'])

class RPCCoeffs(NamedTuple):
    height_off: float
    height_scale: float
    lat_off: float
    lat_scale: float
    line_den_coeff: np.array
    line_num_coeff: np.array
    line_off: float
    line_scale: float
    long_off: float
    long_scale: float
    max_lat: float
    max_long: float
    min_lat: float
    min_long: float
    samp_den_coeff: np.array
    samp_num_coeff: np.array
    samp_off: float
    samp_scale: float

@jit(nopython=True)
def lon_lat_alt_to_xy(
    lon: float,
    lat: float,
    alt: float,
    rpcs: RPCCoeffs,
) -> Tuple[float, float]:
    """
    Returns an image pixel coordinate (x, y) corresponding to a provided world coordinate (lon, lat, alt)
    using provided RPC coefficients
    :param lon: The world coordinate longitude
    :param lat: The world coordinate latitude
    :param alt: The world coordinate altitude
    :param rpcs: NamedTuple with RPC coefficients
    :param interp: Interpolation method -- Linear and Nearest are supported
    :return: Pixel coordinate (x,y) as floating point values -- interpolation will be needed to arrive at a
    pixel intensity value!
    """
    # First create the normalized values for lon/lat/alt
    norm_lon = (lon - rpcs.long_off) / rpcs.long_scale
    norm_lat = (lat - rpcs.lat_off) / rpcs.lat_scale
    norm_alt = (alt - rpcs.height_off) / rpcs.height_scale

    # Create the polynomial vector (gets re-used)
    formula = np.array(
        [
            1,
            norm_lon,
            norm_lat,
            norm_alt,
            norm_lon * norm_lat,
            norm_lon * norm_alt,
            norm_lat * norm_alt,
            norm_lon ** 2,
            norm_lat ** 2,
            norm_alt ** 2,
            norm_lat * norm_lon * norm_alt,
            norm_lon ** 3,
            norm_lon * (norm_lat ** 2),
            norm_lon * (norm_alt ** 2),
            (norm_lon ** 2) * norm_lat,
            norm_lat ** 3,
            norm_lat * (norm_alt ** 2),
            (norm_lon ** 2) * norm_alt,
            (norm_lat ** 2) * norm_alt,
            norm_alt ** 3,
        ],
        dtype=np.float32
    )

    # PLUG AND CHUG
    f1 = np.dot(rpcs.samp_num_coeff, formula)
    f2 = np.dot(rpcs.samp_den_coeff, formula)
    f3 = np.dot(rpcs.line_num_coeff, formula)
    f4 = np.dot(rpcs.line_den_coeff, formula)

    if f2 != 0 and f4 != 0:
        samp_number_normed = f1 / f2
        line_number_normed = f3 / f4
    else:
        pass

    # Then denormalize to get the approximate pixel coordinate
    samp_number = samp_number_normed * rpcs.samp_scale + rpcs.samp_off
    line_number = line_number_normed * rpcs.line_scale + rpcs.line_off

    return samp_number, line_number

# @jit(nopython=True, parallel=True, nogil=True)
# def cross_corr(
#     master: np.ndarray,
#     slave: np.ndarray,
#     win: int,
# ) -> np.array:
#
#     width:
#
#
#     cc_img = np.zeros(width * height)
#
#     #i = np.linspace(
#     #j =
#
#     return cc_img


@jit(nopython=True, parallel=True, nogil=True)
def make_ortho(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    width: int,
    height: int,
    gsd: float,
    source: np.ndarray,
    rpcs: RPCCoeffs,
    dem: np.ndarray,
    dem_geot: np.array,
) -> Tuple[np.array, np.array, float, float, float]:
    """
    Produces an orthorectified image given a
    :param x1: upper left x
    :param x2: lower right x
    :param y1: upper left y
    :param y2: lower right y
    :param width: num pixels for the width of the resulting orthorectified image (height will be set by aspect ratio)
    :param source: the original raw image
    :param rpcs: rational polynomial coefficients object
    :param dem: dem tile corresponding to the image location
    :param dem_geot: dem geo transform (affine transform)
    :param interp: Interpolation type, 0 == nearest neighbor, 1 == bilinear
    :return: a Tuple containing the ortho'd image, the ground sampling distance in degrees,
    and the upper left coordinate
    """

    cols = np.linspace(x1, x2, width)
    #print(cols)
    #gsd = abs(cols[0] - cols[1])
    #print(gsd)
    #height = int(abs(y1 - y2) / gsd)
    #print(height)
    rows = np.linspace(y1, y2, height)
    #print(rows)
    ortho = np.zeros(width * height)

    ortho_dem = np.zeros(width * height)

    for i in prange(len(cols)):
        lon = cols[i]

        for j in prange(len(rows)):
            lat = rows[j]
            dem_x = int((lon - dem_geot[0]) / dem_geot[1])
            dem_y = int((lat - dem_geot[3]) / dem_geot[5])

            if dem_x < 0 or dem_x > dem.shape[1] - 1 or dem_y < 0 or dem_y > dem.shape[0] - 1:
                # Numba will segfault if I don't catch this... AND it will fail to compile
                # if I try to include a useful message with "dem_x" and "dem_y" :( :( :(
                raise IndexError("DEM indices out of bounds")
            altitude = linear_interp(dem_x, dem_y, dem.reshape(-1), dem.shape[1])

            x, y = lon_lat_alt_to_xy(lon, lat, altitude, rpcs)

            if 1 <= x < source.shape[1] - 1 and 1 <= y < source.shape[0] - 1:
                idx = i * height + j
                result = linear_interp(x, y, source.reshape(-1), source.shape[1])
                ortho[idx] = result
                ortho_dem[idx] = altitude
            else:
                pass

    return ortho.reshape(width, height).transpose(), ortho_dem.reshape(width, height).transpose(), gsd, x1, y1


@jit(nopython=True)
def linear_interp(x: float, y: float, source: np.ndarray, source_height: int) -> int:
    x_floor = math.floor(x)
    x_ceil = math.ceil(x)
    y_floor = math.floor(y)
    y_ceil = math.ceil(y)

    x_frac = x - x_floor
    y_frac = y - y_floor

    ul_index = x_floor + source_height * y_floor
    ur_index = x_ceil + source_height * y_floor
    lr_index = x_ceil + source_height * y_ceil
    ll_index = x_floor + source_height * y_ceil

    ul = source[ul_index]
    ur = source[ur_index]
    lr = source[lr_index]
    ll = source[ll_index]

    upper_x = (1 - x_frac) * ul + x_frac * ur
    lower_x = (1 - x_frac) * ll + x_frac * lr
    upper_y = (1 - y_frac) * ul + y_frac * ll
    lower_y = (1 - y_frac) * ur + y_frac * lr
    final_value = (upper_x + lower_x + upper_y + lower_y) / 4

    return int(final_value)


@jit(nopython=True)
def area_interp(x: float, y: float, source: np.ndarray, source_height: int) -> int:
    x_floor = math.floor(x)
    x_ceil = math.ceil(x)
    y_floor = math.floor(y)
    y_ceil = math.ceil(y)

    #x_frac = x - x_floor
    #y_frac = y - y_floor

    ul_index = x_floor + source_height * y_floor
    ur_index = x_ceil + source_height * y_floor
    lr_index = x_ceil + source_height * y_ceil
    ll_index = x_floor + source_height * y_ceil

    ul = source[ul_index]
    ur = source[ur_index]
    lr = source[lr_index]
    ll = source[ll_index]

    idx: 0

    final_value = 0.0

    for i in range(x_ceil-x_floor):
        for j in range(y_ceil-y_floor):
            idx = x_floor + i + source_height * (y_floor + j)
            final_value = final_value + source[idx]

    final_value = final_value/(1+(x_ceil-x_floor)*(y_ceil-y_floor))

    return int(final_value)

def unpack_rpc_parameters(dataset: gdal.Dataset) -> RPCCoeffs:
    """
    Returns RPC coefficients collection as a NamedTuple
    when provided with a GDAL dataset if that dataset contains RPCs
    :param dataset: GDAL dataset reference for an image with RPCs
    :return: A NamedTuple containing RPC coefficients and parameters
    """
    rpc_dict = dataset.GetMetadata_Dict("RPC")
    height_off = float(rpc_dict["HEIGHT_OFF"])
    height_scale = float(rpc_dict["HEIGHT_SCALE"])
    lat_off = float(rpc_dict["LAT_OFF"])
    lat_scale = float(rpc_dict["LAT_SCALE"])
    line_den_coeff = np.array(
        [
            float(coeff.strip())
            for coeff in rpc_dict["LINE_DEN_COEFF"].strip().split(" ")
        ],
        dtype=np.float32
    )
    line_num_coeff = np.array(
        [
            float(coeff.strip())
            for coeff in rpc_dict["LINE_NUM_COEFF"].strip().split(" ")
        ],
        dtype=np.float32
    )
    line_off = float(rpc_dict["LINE_OFF"])
    line_scale = float(rpc_dict["LINE_SCALE"])
    long_off = float(rpc_dict["LONG_OFF"])
    long_scale = float(rpc_dict["LONG_SCALE"])
    max_lat = float(rpc_dict["MAX_LAT"])
    max_long = float(rpc_dict["MAX_LONG"])
    min_lat = float(rpc_dict["MIN_LAT"])
    min_long = float(rpc_dict["MIN_LONG"])
    samp_den_coeff = np.array(
        [
            float(coeff.strip())
            for coeff in rpc_dict["SAMP_DEN_COEFF"].strip().split(" ")
        ],
        dtype=np.float32
    )
    samp_num_coeff = np.array(
        [
            float(coeff.strip())
            for coeff in rpc_dict["SAMP_NUM_COEFF"].strip().split(" ")
        ],
        dtype=np.float32
    )
    samp_off = float(rpc_dict["SAMP_OFF"])
    samp_scale = float(rpc_dict["SAMP_SCALE"])

    return RPCCoeffs(
        height_off,
        height_scale,
        lat_off,
        lat_scale,
        line_den_coeff,
        line_num_coeff,
        line_off,
        line_scale,
        long_off,
        long_scale,
        max_lat,
        max_long,
        min_lat,
        min_long,
        samp_den_coeff,
        samp_num_coeff,
        samp_off,
        samp_scale,
    )


def lon_lat_to_pixel(lon: float, lat: float, geot: np.array) -> Tuple[float, float]:
    """
    Returns a pixel coordinate (x,y) as a tuple when provided with a longitude, latitude coordinate
    and a geo_transform (affine transform)
    :param lon: longitude of the world coordinate
    :param lat: latitude of the world coordinate
    :param geot: affine transform parameters in a length 6 np.array
    :return: A pixel coordinate as a Tuple (x, y)
    """
    x = (lon - geot[0]) / geot[1]
    y = (lat - geot[3]) / geot[5]
    return x, y


def coarse_dem(
    min_lon: float,
    min_lat: float,
    degrees_lon: float,
    degrees_lat: float,
    sampling_rate: int = 1,
    output_path: str = "/elevation.dem",
    ellipsoidal: bool = True,
) -> Tuple[np.ndarray, np.array]:
    """
    Load SRTM tiles for a bouding rectangle defined by an upper left point (min_long, min_lat) and a width and height
    in degrees. Optionally, an integer sampling rate greater than 1 may be passed in to downsample the DEM.
    :param min_lon: x component of the upper left corner of the bounding rectangle
    :param min_lat: y component of the upper left corner of the bounding rectangle
    :param degrees_lon: width of the DEM in degrees
    :param degrees_lat: height of the DEM in degrees
    :param sampling_rate: sampling rate
    :param output_path: Path where the dem file is saved
    :param ellipsoidal: if true return an ellipsoidal DEM
    :return: A numpy ndarray with elevation data, followed by the geotransform of the elevation data wrapped in a Tuple
    """

    load_dem(
        min_lon,
        min_lat,
        degrees_lon,
        degrees_lat,
        rate=sampling_rate,
        data_source='AWS', #"NASA"
        output_name=cfg['temporary_dir']+output_path,
    )
    gdal.AllRegister()
    dem_dataset = gdal.Open(cfg['temporary_dir']+output_path)
    elevation_data = dem_dataset.ReadAsArray()
    geo_transform = np.array([*dem_dataset.GetGeoTransform()])

    if ellipsoidal:

        with open(cfg['geoid_path'], 'r'):
            geoid = geo.GeoidPGM(cfg['geoid_path']).height(min_lat, min_lon)
            print("Geoid file loaded: mean value: ", geoid)

        elevation_data = elevation_data + geoid



    return elevation_data, geo_transform

def save_raster_as_geotiff(ortho: np.ndarray, ul_lon: float, ul_lat: float, lr_lon: float, lr_lat: float, filename: str) -> None:

    gsd_lon = -(ul_lon - lr_lon)/ortho.shape[1]
    gsd_lat = -(ul_lat - lr_lat)/ ortho.shape[0]

    io_driver = gdal.GetDriverByName('GTiff')
    ortho_ds = io_driver.Create(filename, ortho.shape[1], ortho.shape[0], 1, gdal.GDT_Float32) #gdal.GDT_Byte
    ortho_ds.SetGeoTransform((ul_lon, gsd_lon, 0, ul_lat, 0, gsd_lat))
    ortho_band = ortho_ds.GetRasterBand(1)
    ortho_band.WriteArray(ortho)
    coordinate_system = osr.SpatialReference()
    coordinate_system.ImportFromEPSG(4326)
    ortho_ds.SetProjection(coordinate_system.ExportToWkt())
    ortho_band.FlushCache()
    # makes sure resources are released...
    ortho_band = None
    ortho_ds = None