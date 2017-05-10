'''
This file loads in data from /data/interim/interim_data.hdf and then
applies a sinusoidal regression model to the temperature data to remove
the yearly predictable variation.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''

import numpy as np
import pandas as pd
import multiprocessing
import datetime
from scipy.optimize import least_squares
from src.conf import HDF_INTERIM_FILE, LOCATIONS_KEY, TEMPERATURE_TS_ROOT

f_yr = 0.00273515063053  # Frequency of yearly variation in 1/days


# Error function for sinusoidal regression
def err_lstsqr(theta, f, x, t):  # No frequency optimization
    a, phi = theta
    s = a * np.sin(2 * np.pi * f * t + phi)
    return x - s


def sinregress_to_hdf(key: str):
    '''
    Loads in a pandas dataframe from the key location in the hdf store
    given by hdf_path.  We then regress the T series on a sinusoid with
    a period of 1 yr.  We subtract this from T and store it back in
    a 'T_sinregress' column.  This series will be centered as well.
    '''
    with pd.HDFStore(ghdf_path, mode='r') as hdf:
        D = hdf[key]  # Read D from disk
    t = D.index
    T = D['T'].values
    unix_birth = datetime.datetime(1970, 1, 1)

    def time_in_days(t):  # Convert time to raw numbers
        # 86400 = datetime.timedelta(days=1).total_seconds()
        return (t - unix_birth).total_seconds() / 86400

    t_days = np.fromiter(map(time_in_days, t), np.float64)
    res = least_squares(err_lstsqr, (1., 0.), method='lm', verbose=0,
                        kwargs={'f': f_yr, 'x': T, 't': t_days})
    a, phi = res.x
    print('a:', a, 'phi:', phi)  # Watch for odd outliers
    T_hat = a * np.sin(2 * np.pi * f_yr * t_days + phi)
    T_sinregress = T - T_hat
    T_sinregress = T_sinregress - np.mean(T_sinregress)
    D['T_sinregress'] = T_sinregress

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
    hdf_path = HDF_INTERIM_FILE

    # Get location data
    D_loc = pd.read_hdf(hdf_path, key=LOCATIONS_KEY)

    # We have a cpu bound task
    hdf_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=5,
                                initializer=init_lock,
                                initargs=(hdf_lock,
                                          hdf_path),
                                maxtasksperchild=4)
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/wban_'

    pool.map(sinregress_to_hdf, (hdf_group + row['WBAN'] + '/D' for i, row in
                                 D_loc.iterrows()))
    pool.close()
    pool.join()
    return


if __name__ == '__main__':
    main()
