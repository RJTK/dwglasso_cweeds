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
LOCATIONS_ROOT = 'locations'
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
    date_parser = lambda d: datetime.datetime(int(d[0:4]), int(d[4:6]),
                                              int(d[6:8]), int(d[8:10]) - 1)
    D = pd.read_fwf(file_path, colspecs = WY2_cols,
                    header = None, names = WY2_col_names,
                    parse_dates = ['Time'], date_parser = date_parser,
                    converters = {'T_flag' : convert_Tflag})
    D.loc[:, 'T'] = D.loc[:, 'T'] / 10
    D.loc[:, 'Time'] = D.loc[:, 'Time'] + time_correction
    D.index = pd.DatetimeIndex(D.loc[:, 'Time'])
    del D['Time']
    return D

# Temperature data is stored as follows:
#   /<TEMPERATE_TS_ROOT>/wban_<WBAN>.Name <-- Name of location
#   /<TEMPERATE_TS_ROOT>/wban_<WBAN>.WBAN <-- WBAN of location
#   /<TEMPERATE_TS_ROOT>/wban_<WBAN>.lat <-- latitude of station
#   /<TEMPERATE_TS_ROOT>/wban_<WBAN>.lon <-- longitude of station
#   /<TEMPERATE_TS_ROOT>/wban_<WBAN>/D <-- The DataFrame object
#     (read */D with pandas)
def wy2_to_hdf(metadata_row: pd.Series):
    '''
    Loads and processes the wy2 file associated with the given
    metadata row using process_wy2f().  We then write out to
    a file specified by to_hdf.

    NOTE: This is a 'special' function in that it is intended to
    be used by an independent process.  The function init_wy2_to_hdf
    must be called in order to initialize global variables
    'ghdf_lock' and 'ghdf_file'.

    We also replace all the T_flag = 9 rows with np.nan in the T col
    '''
    print('Processing ', metadata_row['Name'])
    #This takes vastly more time than writing out to the hdf database.
    D = process_wy2f(metadata_row['WBAN_file'], metadata_row['time_correction'])
    hdf_group = '/' + TEMPERATURE_TS_ROOT + '/'
    hdf_record = 'wban_' + str(metadata_row['WBAN'])
    node_key = hdf_group + hdf_record
    D_key = hdf_group + hdf_record + '/D'
    with ghdf_lock:
        D.to_hdf(ghdf_file, key = D_key)
        with tables.open_file(ghdf_file, mode = 'r+') as hdf:
            hdf.set_node_attr(node_key, 'Name', metadata_row['Name'])
            hdf.set_node_attr(node_key, 'WBAN', metadata_row['WBAN'])
            hdf.set_node_attr(node_key, 'lat', metadata_row['lat'])
            hdf.set_node_attr(node_key, 'lon', metadata_row['lon'])    
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

    #Create an h5f database
    h5f = tables.open_file(hdf_file, mode = 'w')
    h5f.create_group('/', TEMPERATURE_TS_ROOT)
    h5f.close()

    #Store the location data in the database as well
    D_loc.to_hdf(hdf_file, '/' + LOCATIONS_ROOT + '/D')

    hdf_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer = init_wy2_to_hdf,
                                initargs = (hdf_lock, hdf_file))
    pool.map(wy2_to_hdf, (row for i, row in D_loc.iterrows()))
    pool.close()
    pool.join()

    return

if __name__ == '__main__':
    main()
