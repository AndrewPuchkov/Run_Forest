import rasterio
import numpy as np
#img_path = r"C:\Users\USER33\University\Run_Forest\test_segmentation\global_monthly_2018_09_mosaic_L15-1630E-0988N_6522_4239_13.tif"

#with rasterio.open(img_path) as src:
#    print(src.count)



with rasterio.open(r"D:\run_forest\data\jpg_data\masks\0_mask.jpg") as src:
    mask = src.read(1)

print(np.unique(mask))