import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import data_pipeline

data = data_pipeline.preprocess_LKOS_data(filename="Fallzahlen 28.04.21.xlsx").to_numpy()
number_days = 20
data = data[0:data.shape[0]-2, 0:]
data = np.sum(data, axis=1)


coordinates = [(13,8),(7,2),(8,0),(10,1),(9,7),(10,5),(13,9),(6,11),(11,1),(8,4),(7,0),(6,4),(6,5),(9,2),(13,4),(11,8),(7,9),(6,18),(3,19),(4,17),(6,19),(6,14),(4,15),(5,15),(3,15),(6,16),(3,16),(6,13),(1,17),(1,16),(0,14),(3,13),(3,11),(1,11),(7,6)]
heatmap = np.zeros((14,20))
for index, coordinate in enumerate(coordinates):
    heatmap[coordinate[0], coordinate[1]] = data[index]
plt.imshow(heatmap.T, cmap='hot', origin='lower')
plt.show()


grid_x, grid_y = np.mgrid[0:14, 0:20]
heatmap_interpolated = griddata(coordinates, data, (grid_x, grid_y), method='linear')
heatmap_interpolated[np.isnan(heatmap_interpolated)] = 0
plt.imshow(heatmap_interpolated.T, cmap='hot', origin='lower')
plt.show()

