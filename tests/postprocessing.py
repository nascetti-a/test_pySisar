import numpy as np

import rasterio
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter

file_DSM = "/Volumes/Samsung_T5/DEM_Extraction_pyDATE/Cushing/pydateNCC/finale_dem.tiff"

file_ortho = "/Volumes/Samsung_T5/DEM_Extraction_pyDATE/Cushing/pydateNCC"

src = rasterio.open(file_DSM)

result = gaussian_filter(src.read(1), sigma=5)

pyplot.imshow(result, cmap='pink')
pyplot.show()

with rasterio.open("/Volumes/Samsung_T5/DEM_Extraction_pyDATE/Cushing/pydateNCC/" + 'smooth.tif', 'w', **src.profile) as dst:
    dst.write(result, 1)