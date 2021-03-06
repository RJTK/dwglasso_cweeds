'''
This file loads in data from /data/interim/interim_data.hdf and then
both centers the temperature data and adds in a dT column of temperature
differences.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''

import pandas as pd
import sys
from src.conf import HDF_INTERIM_FILE, LOCATIONS_KEY, TEMPERATURE_TS_ROOT
from scipy.stats import uniform


def temp_diff_to_hdf(hdf_path: str, key: str):
    '''
    Loads in a pandas dataframe from the key location in the hdf store
    given by hdf_path.  We then truncate the series so that it does
    not begin or end with unobserved data, we center the 'T' column,
    and we add a 'dT' column consisting of the first differences of
    the 'T' column.  This series will also be centered.
    '''
    with pd.HDFStore(hdf_path, mode='r') as hdf:
        D = hdf[key]  # Read D from disk
    # Trucate so that we don't start or end with unobserved data
    t = D.index  # The times of observation
    t_obs = t[D['T_flag'] != -1]
    D = D[t_obs[0]:t_obs[-1]]  # Truncate

    # Center the temperature series
    T = D['T']
    mu = T.mean()
    T = T - mu
    D.loc[:, 'T'] = T

    # Get the differences.  Note that dT[0] = np.nan
    dT = T.diff()
    # After about 1978 the data discretization is within 0.1degrees C,
    # I dither the data so as to prevent any numerical issues resulting
    # from this discretization.
    dT = dT + uniform.rvs(loc=-0.5, scale=1.0, size=len(dT))
    dT = dT - dT.mean()  # Ensure to center the differences too
    D['dT'] = dT

    # Open the database and write out the result.
    D.to_hdf(hdf_path, key=key)
    return


def main():
    hdf_path = HDF_INTERIM_FILE

    # This task is mostly io bound, so there is no reason to
    # do anything in parallel as in interpolate_data.py

    # Get the location data
    D_loc = pd.read_hdf(hdf_path, key=LOCATIONS_KEY)
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/wban_'
    N = len(D_loc)
    for i, row in D_loc.iterrows():
        print('Processing record: ', i, '/', N, end='\r')
        sys.stdout.flush()
        temp_diff_to_hdf(hdf_path, hdf_group + row['WBAN'] + '/D')
    return


if __name__ == '__main__':
    main()
