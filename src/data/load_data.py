'''
Loads in the data from all the wy2 files in /data/raw/.  This must
be run after we have processed the locations metadata.  We read this
metadata from <LOC_PKL_FILE> and then store all the data into an hdf
database <HDF_FILE> having <TEMPERATURE_TS_ROOT> as the root for the
temperature data

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''

import os
import logging
import datetime
import multiprocessing
import tables
import pandas as pd

LOC_PKL_FILE = 'locations.pkl' #Name of the locations metadata file
HDF_FILE = 'interim_data.hdf' #Name of the hdf file for ts data
TEMPERATURE_TS_ROOT = 'temperature_ts' #Name of the temperature key in hdf

WY2_cols = [(6, 16), (91, 95), (95, 96)] #Time, temperature, temp flag
WY2_col_names = ['Time', 'T', 'T_flag'] #Flags indicate missing or estimated data

def process_wy2f(file_path: str, time_correction: datetime.timedelta):
    '''
    Processes and returns a wy2 file, correcting for time,
    converting flags to numerical values, and taking the
    temperature to degrees C.

    The file_path and time_correction should be obtained
    from the LOC_PKL_FILE location metadata.

    The T_flags are given as follows:
    #0: Observed
    #-1: Missing data
    #1: Algorithmically adjusted
    #2: Hand estimate
    #3: Interpolated
    #4: Determined from a model
    #5: Derived
    '''
    Tflags = {'': 0, '9': -1, 'A': 1, 'E': 2, 'I': 3, 'M': 4, 'Q': 5}
    convert_Tflag = lambda tf: Tflags[tf]

    #Parsing the date takes by far the most time
    td_1hr = datetime.timedelta(hours = 1)
    date_parser = lambda d: pd.to_datetime(str(int(d) - 1),
                                           format = '%Y%m%d%H') + td_1hr
    D = pd.read_fwf(file_path, colspecs = WY2_cols,
                    header = None, names = WY2_col_names,
                    parse_dates = ['Time'], date_parser = date_parser,
                    converters = {'T_flag' : convert_Tflag})
    D.loc[:, 'T'] = D.loc[:, 'T'] / 10
    D.loc[:, 'Time'] = D.loc[:, 'Time'] + time_correction
    return D

def wy2_to_hdf(metadata_row: pd.Series):
    '''
    Loads and processes the wy2 file associated with the given
    metadata row using process_wy2f().  We then write out to
    a file specified by to_hdf.

    NOTE: This is a 'special' function in that it is intended to
    be used by an independent process.  The function init_wy2_to_hdf
    must be called in order to initialize global variables
    'ghdf_lock' and 'ghdf_file'.
    '''
    print('Processing ', metadata_row['Name'])
    #This takes vastly more time than writing out to the hdf database.
    D = process_wy2f(metadata_row['WBAN_file'], metadata_row['time_correction'])
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/'
    hdf_record = 'wban_' + str(metadata_row['WBAN'])
    key = hdf_group + hdf_record
    with ghdf_lock:
        h5f = tables.open_file(ghdf_file, mode = 'r+')
        h5f.create_group(hdf_group, hdf_record)
        h5f.set_node_attr(key, 'Name', metadata_row['Name'])
        h5f.set_node_attr(key, 'WBAN', metadata_row['WBAN'])
        h5f.set_node_attr(key, 'lat', metadata_row['lat'])
        h5f.set_node_attr(key, 'long', metadata_row['long'])    
        h5f.close()
        D.to_hdf(ghdf_file, key = key)
    return

def init_wy2_to_hdf(lock: multiprocessing.Lock, hdf_file: str):
    '''
    The init function for wy2_to_hdf.
    This needs to be run before we use wy2_to_hdf
    '''
    global ghdf_lock
    global ghdf_file
    ghdf_lock = lock
    ghdf_file = hdf_file
    return

def main():
    '''Program entry point'''
    cwd = os.getcwd()
    hdf_file = cwd + '/data/interim/' + HDF_FILE
    #Load location metadata
    D_loc = pd.read_pickle(cwd + '/data/interim/' + LOC_PKL_FILE)

    #Create an h5f database for the time series data
    h5f = tables.open_file(hdf_file, mode = 'w')
    h5f.create_group('/', TEMPERATURE_TS_ROOT)
    h5f.close()

    hdf_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer = init_wy2_to_hdf,
                                initargs = (hdf_lock, hdf_file))
    pool.map(wy2_to_hdf, (row for i, row in D_loc.iterrows()))
    pool.close()
    pool.join()
    return

if __name__ == '__main__':
    main()
