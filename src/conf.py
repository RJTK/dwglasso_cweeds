'''
This is the config file for the code in src/.  Essentially it
holds things like file and variable names.
'''

# The folder locations of the below files are specified by the
# cookie cutter data science format and are hardcoded into the code.
# I'm not entirely sure that that was the best way to go about it,
# but thats how it is for now.

import os
cwd = os.getcwd()  # Current working directory

# Directories continaing data
RAW_DATA_DIR = cwd + '/data/raw/'
INTERIM_DATA_DIR = cwd + '/data/interim/'
PROCESSED_DATA_DIR = cwd + '/data/processed/'

# Path of initial locations text file
LOC_DATA_FILE = RAW_DATA_DIR + 'locations.txt'

# Path to pickle location data
LOC_PKL_FILE = INTERIM_DATA_DIR + 'locations.pkl'

# Path to interim data HDFStore
HDF_INTERIM_FILE = INTERIM_DATA_DIR + 'interim_data.hdf'

LOCATIONS_ROOT = 'locations'  # The root node of location data in hdf
TEMPERATURE_TS_ROOT = 'temperature_ts'  # Name of the temperature key in hdf
INIT_YEAR = 1980  # The initial year for final dataset
FINAL_YEAR = 1990  # The final year for final dataset
