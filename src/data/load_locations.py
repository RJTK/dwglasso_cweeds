'''
This file loads in the fixed width file containing the information
about each weather station's lat/long coordinates as well as the mlong
prime meridian for the local standard time (LST) to universal time
(UTC).  We perform the time correction to UTC, and also find the
path to the time series file for the station.

Location data is stored in /data/interim/<PKL_FILE>.pkl, and loaded
from the file /data/interim/<LOC_DATA_FILE>.pkl.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import os
import datetime

import pandas as pd

col_names = ['Name', 'WBAN', 'lat', 'long', 'mlong', 'first_year',
             'last_year']
loc_cols = [(0, 24), (24, 30), (45, 52), (52, 58), (59, 65), (74, 76),
            (77, 79)]

LOC_DATA_FILE = 'locations.txt'
PKL_FILE = 'locations.pkl'

def fix_year(yr: int):
    '''
    The year is specified only with the last 2 digits but, data
    collection started in after 1950 and ended before 2050
    '''
    if yr > 50:
        yr += 1900
    else:
        yr += 2000
    return yr

def time_correction(mlong: float):
    '''
    The time delta to add to an LST time to yield a UTC time,
    given the prime meridian mlong in degrees.
    '''
    return datetime.timedelta(minutes = mlong / 15)

def wban_fname(wban: str):
    '''Convert the WBAN string into the filename we need to look for'''
    #CARE: This will give the files a relative path name.
    #It will only work from the directory of this file.
    cwd = os.getcwd()
    for root, dirs, files in os.walk(cwd + '/data/raw/'):
        for f in files:
            if f.endswith('WY2') and f.startswith(wban):
                return root + '/' + f
    raise ValueError('404 wban %s not found!' % wban)

def main():
    '''Program entry point'''
    cwd = os.getcwd() #Current working directory
    try:
        D = pd.read_fwf(cwd + '/data/raw/' + LOC_DATA_FILE, colspecs = loc_cols,
                        comment = '#', header = None, names = col_names)
    except FileNotFoundError:
        raise FileNotFoundError('The file ' + cwd +
                                '/data/raw/' + LOC_DATA_FILE +
                                ' does not exist.  Was this script ' +
                                'executed by make from the top level?')

    D.loc[:, ['first_year', 'last_year']] =\
        D.loc[:, ['first_year', 'last_year']].applymap(fix_year)
    D['time_correction'] = D.loc[:, 'mlong'].apply(time_correction)
    del D['mlong'] #No longer needed
    D['WBAN_file'] = D['WBAN'].apply(wban_fname)
    D.to_pickle(cwd + '/data/interim/' + PKL_FILE)
    return

if __name__ == '__main__':
    main()
