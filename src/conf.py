'''
This is the config file for the code in src/.  Essentially it
holds things like file and variable names.
'''

# The folder locations of the below files are specified by the
# cookie cutter data science format and are hardcoded into the code.
# I'm not entirely sure that that was the best way to go about it,
# but thats how it is for now.

LOC_DATA_FILE = 'locations.txt'  # Name of initial locations text file
HDF_INTERIM_FILE = 'interim_data.hdf'  # Name of interim hdf file
LOC_PKL_FILE = 'locations.pkl'  # Name of the locations metadata file
LOCATIONS_ROOT = 'locations'  # The root node of location data in hdf
TEMPERATURE_TS_ROOT = 'temperature_ts'  # Name of the temperature key in hdf
INIT_YEAR = 1980  # The initial year for final dataset
FINAL_YEAR = 1990  # The final year for final dataset
