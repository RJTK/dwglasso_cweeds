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
    INIT_YEAR, FINAL_YEAR, FINAL_YEAR_VALIDATE, HDF_FINAL_FILE,\
    X_VALIDATE_FILE_PREFIX


def main():
    '''
    '''
    hdf_interim = pd.HDFStore(HDF_INTERIM_FILE, mode='r')
    hdf_final = pd.HDFStore(HDF_FINAL_FILE, mode='w')

    # Dataframe containing list of the WBANs
    D_loc_interim = hdf_interim[LOCATIONS_KEY]
    N = len(D_loc_interim)

    t0_train = datetime(INIT_YEAR, 1, 1)
    tf_train = datetime(FINAL_YEAR, 12, 31, 23)
    t0_validate = datetime(FINAL_YEAR + 1, 1, 1)
    tf_validate = datetime(FINAL_YEAR_VALIDATE, 12, 31, 23)

    t_train = pd.date_range(t0_train, tf_train, freq='H')
    t_validate = pd.date_range(t0_validate, tf_validate, freq='H')
    t = pd.date_range(t0_train, tf_validate, freq='H')
    assert len(t) == len(t_train) + len(t_validate), 'Issue with t indices!'

    n_usable = 0  # The number of usable series
    D_loc_final = pd.DataFrame()  # The final location data
    for i, row in D_loc_interim.iterrows():
        wban = row['WBAN']
        print('Processing (', wban, ') ', i + 1, '/', N, end='\r')
        k = TEMPERATURE_TS_ROOT + '/wban_' + wban + '/D'
        D = hdf_interim[k]
        tk = D.index
        if tk[0] <= t0_train and tk[-1] >= tf_validate:
            n_usable += 1
            # Raw temperature data
            T_train = D.loc[t_train, 'T'].values
            T_validate = D.loc[t_validate, 'T'].values
            T_train = T_train - np.mean(T_train)
            T_validate = T_validate - np.mean(T_validate)

            # Temperature difference data
            dT_train = D.loc[t_train, 'dT'].values
            dT_validate = D.loc[t_validate, 'dT'].values
            dT_train = dT_train - np.mean(dT_train)
            dT_validate = dT_validate - np.mean(dT_validate)
            Dk = pd.DataFrame(data={'T': T_train, 'dT': dT_train},
                              index=t_train)
            hdf_final[k] = Dk
            D_loc_final = D_loc_final.append(row)

            try:
                XT_validate = np.vstack((XT_validate,
                                         T_validate[np.newaxis, :]))
                XdT_validate = np.vstack((XdT_validate,
                                          dT_validate[np.newaxis, :]))
            except NameError:
                XT_validate = T_validate[np.newaxis, :]
                XdT_validate = dT_validate[np.newaxis, :]

    print('\nNumber of series in final dataset: ', n_usable)
    hdf_interim.close()

    hdf_final[LOCATIONS_KEY] = D_loc_final
    hdf_final.close()

    np.save(X_VALIDATE_FILE_PREFIX + '_T', XT_validate)
    np.save(X_VALIDATE_FILE_PREFIX + '_dT', XdT_validate)
    return


if __name__ == '__main__':
    main()
