'''
This file loads in data from /data/interim/interim_data.hdf and then
interpolates the missing data for each temperature time series.  We
interpolate with the 'pchip' method.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import pandas as pd
import numpy as np
import os
import multiprocessing

DATA_DIR = '../data/interim/'
HDF_FILE = 'interim_data.hdf'

LOC_PKL_FILE = 'locations.pkl'  # Name of the locations metadata file
LOCATIONS_ROOT = 'locations'
HDF_FILE = 'interim_data.hdf'  # Name of the hdf file for ts data
TEMPERATURE_TS_ROOT = 'temperature_ts'  # Name of the temperature key in hdf


def interpolate_to_hdf(key: str, method='pchip', order=None):
    '''
    Loads in a pandas dataframe from the key location in the hdf store
    given by hdf_path then interpolates the 'T' column using the
    specified method and writes back to the key location.  We also look
    at the T_flag column to determine which columns to interpolate.
    '''
    print('Processing key ', key)
    with pd.HDFStore(ghdf_path, mode='r') as hdf:
        D = hdf[key]
    i_miss = D.index[D['T_flag'] == -1]
    D.loc[i_miss, 'T'] = np.nan
    D.loc[:, 'T'] = D.loc[:, 'T'].interpolate(method=method,
                                              order=order,
                                              limit_direction='forward',
                                              axis=0)
    with ghdf_lock:
        D.to_hdf(ghdf_path, key=key)
    return


def init_lock(lock: multiprocessing.Lock, hdf_path: str):
    '''
    Initializes the global lock used by interpolate_to_hdf
    '''
    global ghdf_lock
    global ghdf_path
    ghdf_lock = lock
    ghdf_path = hdf_path
    return


def main():
    cwd = os.getcwd()
    hdf_path = cwd + '/data/interim/' + HDF_FILE
    loc_key = '/' + LOCATIONS_ROOT + '/D'

    # Get the location data
    D_loc = pd.read_hdf(hdf_path, key=loc_key)

    # See the interpolation notebook
    # This task is CPU bound by a long shot
    hdf_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init_lock,
                                initargs=(hdf_lock,
                                          hdf_path))
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/wban_'
    pool.map(interpolate_to_hdf, (hdf_group + row['WBAN'] + '/D' for i, row in
                                  D_loc.iterrows()))
    pool.close()
    pool.join()
    return


if __name__ == '__main__':
    main()
