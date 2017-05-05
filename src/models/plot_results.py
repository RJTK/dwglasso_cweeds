'''
Plots all the results of the dwglasso analysis on a map of Canada.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
from dwglasso import dwglasso
import matplotlib as mpl; mpl.use('TkAgg')
from matplotlib import pyplot as plt
from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from src.conf import CANADA_SHAPE, HDF_FINAL_FILE, LOCATIONS_KEY, MAX_P,\
    P_LAG, ZZT_FILE_PREFIX, YZT_FILE_PREFIX


def main():
    p = P_LAG
    assert p <= MAX_P and p >= 1, 'p must be in (1, MAX_P)!'
    ZZT = np.load(ZZT_FILE_PREFIX + str(p) + '_T' + '.npy')
    YZT = np.load(YZT_FILE_PREFIX + str(p) + '_T' + '.npy')

    # plt.imshow(ZZT)
    # plt.colorbar()
    # plt.show()

    # Run dwglasso
    B_hat = dwglasso(ZZT, YZT, p, lmbda=261.225, alpha=0.25, tol=1e-11,
                     mu=0.01, max_iter=200, sigma=2.5, delta=0.1,
                     ret_B_err=False)

    # plt.imshow(B_hat)
    # plt.colorbar()
    # plt.title('DWGLASSO Result')
    # plt.show()

    n = B_hat.shape[0]
    assert B_hat.shape[1] // n == p, 'Issue with matrix sizes!'

    # Get the actual causality matrix
    G = np.abs(sum([B_hat[:, tau * n:(tau + 1) * n] for tau in range(p)]).T)
    G = G > 0  # The Granger-causality graph
    G = G - np.diag(np.diag(G))  # Zero the diagonal (no looping edges)

    print('Num edges: ', np.sum(G), '/', ((n * p) * (n * p) + 1) // 2)

    # plt.imshow(G)
    # plt.title('Granger-causality Graph')
    # plt.show()

    # Plot the map of canada
    canada = gpd.read_file(CANADA_SHAPE)
    del canada['NOM']  # Drop French names
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Canada')
    ax.axes.get_yaxis().set_ticks([])  # Remove the ticks
    ax.axes.get_xaxis().set_ticks([])
    canada.plot(ax=ax, color='white')

    hdf_final = pd.HDFStore(HDF_FINAL_FILE, mode='r')
    D_loc = hdf_final[LOCATIONS_KEY]

    # Plot on the map all the station locations
    station_geometry = [Point(latlon) for latlon in zip(-D_loc['lon'],
                                                        D_loc['lat'])]
    stations = gpd.GeoDataFrame(pd.DataFrame(D_loc['Name'].values,
                                             columns=['NAME']),
                                geometry=station_geometry,
                                crs={'init': 'epsg:4326'})
    stations.to_crs(crs=canada.crs, inplace=True)
    stations.plot(ax=ax, marker='*', markersize=6, color='red')

    # Plot arrows corresponding to the causality structure
    arrow_geometry = []
    connected_stations = []
    for i, j in combinations_with_replacement(stations.index, 2):
        if G[i, j] or G[j, i]:
            connected_stations.append('%d --- %d' % (i, j))
            arrow_geometry.append(LineString((station_geometry[i],
                                             station_geometry[j])))

    arrows = gpd.GeoDataFrame(pd.DataFrame(connected_stations),
                              geometry=arrow_geometry,
                              crs={'init': 'epsg:4326'})
    arrows.to_crs(crs=canada.crs, inplace=True)
    arrows.plot(ax=ax, color='black')
    plt.show()
    return


if __name__ == '__main__':
    main()
