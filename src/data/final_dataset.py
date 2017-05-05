'''
Creates the final dataset to which we will apply dwglasso.  This
is important because the time series of available measurements is not
consistent accross every location.  There are over 200 stations, but
not all of them have temperature observations over the same periods.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import pandas as pd
import numpy as np
from datetime import datetime
from src.conf import HDF_INTERIM_FILE, LOCATIONS_KEY, TEMPERATURE_TS_ROOT,\
    INIT_YEAR, FINAL_YEAR, HDF_FINAL_FILE


def main():
    '''
    '''
    hdf_interim = pd.HDFStore(HDF_INTERIM_FILE, mode='r')
    hdf_final = pd.HDFStore(HDF_FINAL_FILE, mode='w')

    # Dataframe containing list of the WBANs
    D_loc_interim = hdf_interim[LOCATIONS_KEY]
    N = len(D_loc_interim)

    t0 = datetime(INIT_YEAR, 1, 1)
    tf = datetime(FINAL_YEAR, 12, 31, 23)
    t = pd.date_range(t0, tf, freq='H')

    n = 0  # The number of usable series
    D_loc_final = pd.DataFrame()  # The final location data
    for i, row in D_loc_interim.iterrows():
        wban = row['WBAN']
        print('Processing (', wban, ') ', i + 1, '/', N, end='\r')
        k = TEMPERATURE_TS_ROOT + '/wban_' + wban + '/D'
        D = hdf_interim[k]
        tk = D.index
        if tk[0] <= t0 and tk[-1] >= tf:
            n += 1
            T = D.loc[t, 'T'].values
            T = T - np.mean(T)
            dT = D.loc[t, 'dT'].values
            dT = dT - np.mean(dT)  # Ensure the data is centered
            Dk = pd.DataFrame(data={'T': T, 'dT': dT}, index=t)
            hdf_final[k] = Dk
            D_loc_final = D_loc_final.append(row)
    print('\nNumber of series in final dataset: ', n)
    hdf_interim.close()

    hdf_final[LOCATIONS_KEY] = D_loc_final
    hdf_final.close()
    return


if __name__ == '__main__':
    main()
