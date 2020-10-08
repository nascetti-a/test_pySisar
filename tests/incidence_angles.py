import cv2
import numpy as np
import pandas as pd
import rpcm
import pygeodesy as geod

import plotly.express as exp

from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from pySISAR import retrieve_dem


#root = "../../../../Volumes/Samsung_T5/Satellite_Imagery/ICEYE/Cushing/"

root = "../../../../../Volumes/Samsung_T5/Satellite_Imagery/Pleiades_Trento/"


px = 300
py = 300

#img1 = cv2.imread(root + './Lee_Filter/Cushing/subset_ICEYE_X4_SLC_SL_27170_20200420T212130_Spk_Cnv.tif', cv2.IMREAD_UNCHANGED)
#img2 = cv2.imread(root + './Lee_Filter/Cushing/subset_ICEYE_X5_SLC_SL_27176_20200508T211604_Spk_Cnv.tif', cv2.IMREAD_UNCHANGED)

#img2 = cv2.imread(root + './Lee_Filter/Cushing/ICEYE_X5_SLC_SL_27177_20200525T205123.h5_Spk_Cnv.tif', cv2.IMREAD_UNCHANGED)
#img1 = cv2.imread(root + './Lee_Filter/Cushing/ICEYE_X4_SLC_SL_27172_20200505T204745.h5_Spk_Cnv.tif', cv2.IMREAD_UNCHANGED)


output_name = "27176_27172_20"

img1 = cv2.imread(root + 'IMG_PHR1A_P_201208281022063_SEN_1299536101-003_R1C1.TIF', cv2.IMREAD_ANYDEPTH )
img2 = cv2.imread(root + 'IMG_PHR1A_P_201208281022159_SEN_1299536101-002_R1C1.TIF', cv2.IMREAD_ANYDEPTH )


print(type(img1), img1.shape)


#rpc_img1 = rpcm.rpc_from_rpc_file(root + './Lee_Filter/Cushing/ICEYE_X4_SLC_SL_27170_20200420T212130_RPC.txt')
#rpc_img2 = rpcm.rpc_from_rpc_file(root + './Lee_Filter/Cushing/ICEYE_X5_SLC_SL_27176_20200508T211604_RPC.txt')


#rpc_img2 = rpcm.rpc_from_rpc_file(root + './Lee_Filter/Cushing/ICEYE_X5_SLC_SL_27177_20200525T205123_RPC.txt')
#rpc_img1 = rpcm.rpc_from_rpc_file(root + './Lee_Filter/Cushing/ICEYE_X4_SLC_SL_27172_20200505T204745_RPC.txt')


rpc_img1 = rpcm.rpc_from_rpc_file(root + 'RPC_PHR1A_P_201208281022063_SEN_1299536101-003.XML')
rpc_img2 = rpcm.rpc_from_rpc_file(root + 'RPC_PHR1A_P_201208281022159_SEN_1299536101-002.XML')





# Define the AOI the GSD for CUSHING ICEYE
#dem = retrieve_dem(-96.8, 36.0, 0.2,0.3)

#UL_Lon, UL_Lat = (-96.7750, 35.9750)
#LR_Lon, LR_Lat = (-96.7200, 35.9425)
#gsd = 2.0

#Define the AOI the GSD


dem = retrieve_dem(11, 46.3, 0.3,0.3)

UL_Lon, UL_Lat = (11.120, 46.065)
LR_Lon, LR_Lat = (11.140, 46.045)
gsd = 0.5


#Compute the corresponding ortho grid dimensions

UL_UTM = geod.toUtm8(UL_Lat, UL_Lon)
LR_UTM = geod.toUtm8(LR_Lat, LR_Lon)
width = int((LR_UTM.easting - UL_UTM.easting) / gsd)
height = int((UL_UTM.northing - LR_UTM.northing) / gsd)


#Incidence Angles maps

from matplotlib.ticker import FormatStrFormatter

N_grid = 25

mat_azimuth = np.zeros((N_grid, N_grid))
mat_zenith = np.zeros((N_grid, N_grid))

cols = np.linspace(UL_Lon, LR_Lon, N_grid)
rows = np.linspace(LR_Lat, UL_Lat, N_grid)

BBox = [UL_Lon, LR_Lon, LR_Lat, UL_Lat]


for i, lon  in enumerate(cols):
    for j, lat in enumerate(rows):
        mat_zenith[i, j], mat_azimuth[i, j] = rpc_img2.incidence_angles(lon, lat, 200)
        #print(mat_zenith[i, j], mat_azimuth[i, j])
        #print(rpc_img2.incidence_angles(lon, lat, 200))


def plot_image_map(img, BBox, title='Figure'):
    fig, ax = plt.subplots()
    pos = ax.imshow(img, extent=BBox)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.colorbar(pos, ax=ax)


plot_image_map(mat_zenith, BBox, "Zenith Angle Map")

plot_image_map(mat_azimuth, BBox, "Azimuth Angle Map")

plt.show()


f_z = RegularGridInterpolator((cols, rows), mat_zenith, method='linear')

f_a = RegularGridInterpolator((cols, rows), mat_azimuth, method='linear')

pts = np.random.random((100, 2))

d_zenith = []
d_azimuth = []

df_pts = pd.DataFrame({'lat': [], "lon": [], 'd_a': [], 'd_z': []})

for p in pts:
    p_lon =  p[0]*(LR_Lon-UL_Lon) + UL_Lon
    p_lat =  p[1]*(UL_Lat-LR_Lat) + LR_Lat
    inter_val_z = f_z((p_lon, p_lat))
    inter_val_a = f_a((p_lon, p_lat))

    rpc_val = rpc_img2.incidence_angles(p_lon, p_lat, 200)

    d_zenith.append(rpc_val[0] - inter_val_z)
    #print(rpc_val[0], inter_val_z, rpc_val[0] - inter_val_z)

    df_pts = df_pts.append({'lat': p_lat, "lon": p_lon, 'd_a': rpc_val[1] - inter_val_a, 'd_a': rpc_val[0] - inter_val_z},
                  ignore_index=True)

    d_azimuth.append(rpc_val[1] - inter_val_a)
    print(p_lon, p_lat, rpc_val, inter_val_z, inter_val_a)


print("Mean    Azimith: ", np.mean(d_azimuth))
print("Std.Dev Azimith: ", np.std(d_azimuth))

print("Mean    Zenith: ", np.mean(d_zenith))
print("Std.Dev Zenith: ", np.std(d_zenith))

print("11.121, 46.046: ", rpc_img2.incidence_angles(11.121, 46.046, 200), f_z((11.121,46.046)), f_a((11.121,46.046)))

print("11.139, 46.064:", rpc_img2.incidence_angles(11.140, 46.065, 200) , f_z((11.140,46.065)), f_a((11.140,46.065)))

print(d_azimuth)

print(df_pts.head())

fig = exp.scatter_mapbox(df_pts, lat="lat", lon="lon", hover_data=["d_a", "lat", "lon"], zoom=10,
                        height=550, width=650)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title="Azimuth_Residuals", margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()


plt.show()


