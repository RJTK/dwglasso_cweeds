'''
This file loads in data from /data/interim/interim_data.hdf and then
both centers the temperature data and adds in a dT column of temperature
differences.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''

import pandas as pd
import numpy as np
import os
import sys
import multiprocessing

DATA_DIR = '../data/interim/'
HDF_FILE = 'interim_data.hdf'

LOC_PKL_FILE = 'locations.pkl'  # Name of the locations metadata file
LOCATIONS_ROOT = 'locations'
HDF_FILE = 'interim_data.hdf'  # Name of the hdf file for ts data
TEMPERATURE_TS_ROOT = 'temperature_ts'  # Name of the temperature key in hdf


def temp_diff_to_hdf(hdf_path, key: str):
    '''
    Loads in a pandas dataframe from the key location in the hdf store
    given by hdf_path.  We then truncate the series so that it does
    not begin or end with unobserved data, we center the 'T' column,
    and we add a 'dT' column consisting of the first differences of
    the 'T' column.  This series will also be centered.

    '''
    with pd.HDFStore(hdf_path, mode='r') as hdf:
        D = hdf[key]
    #Do the stuff described in the docstring.
    return


def main():
    cwd = os.getcwd()
    hdf_path = cwd + '/data/interim/' + HDF_FILE
    loc_key = '/' + LOCATIONS_ROOT + '/D'

    # Get the location data
    D_loc = pd.read_hdf(hdf_path, key=loc_key)
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/wban_'
    N = len(D_loc)
    for i, row in D_loc.iterrows():
        print('Processing record: ', i, '/', N, end='\r')
        sys.stdio.flush()
        temp_diff_to_hdf(hdf_path, hdf_group + row['WBAN'] + '/D')
    return


if __name__ == '__main__':
    main()
