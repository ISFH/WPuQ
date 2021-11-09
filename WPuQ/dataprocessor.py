from pathlib import Path
from datetime import datetime as dt
from datetime import date
from datetime import time
import os
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker

from .quarters import pv_objs
from .h5ls import H5ls
from .util import (
    rreplace,
    fill_outliers,
    correct_from_device_parameters,
    harmonize_timestamp)

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


idx = pd.IndexSlice

font = {'family': 'normal',
        'size': 20}

matplotlib.rc('font', **font)


class Dataprocessor():
    '''
    This class processes the data of the WPuQ project and restructures it to
    seperate HDF5 files.

    Parameters
    ----------

    Attributes
    ----------
    pv_objs : list
        A list containing the objects with PV panels installed
    '''

    def __init__(self):
        pass

    def get_pv_objects(self, quarter):
        '''
        Gets the pv objects from a given quarter

        Parameters
        ----------
        quarter : str
            The quarter to get objects from

        Returns
        -------
        None

        '''
        self.pv_objs = pv_objs[quarter]

    def adjust_raw_data(self, profile, corrections, metadata, dset_name):
        '''
        Corrects the raw data

        Parameters
        ----------
        profile : pd.DataFrame
            The profile to correct
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')
        metadata : h5py.AttributeManager
            Dict-like object that contains the metadata
        dset_name : str
            The group path inside the HDF5 file

        Returns
        -------
        profile : pd.DataFrame
            The corrected profile
        memento : dict
            A dictionary with the same keys as corrections' keys. Each entry
            is a pd.Index containing the timestamps of missing data

        '''
        profile_old = profile.copy()
        memento = dict()
        if corrections['device_bounds']:
            lb, ub = (float(x) for x in metadata['measure_range']
                      .decode('utf-8').split(','))
            # feed names
            feed_names = dict()
            feed_names_raw = profile.name.split(':')
            for nfeed in range(len(feed_names_raw)):
                feed_names[nfeed] = feed_names_raw[nfeed]
            # tags
            tag_raw = dset_name.split('_')[0].split('/')
            tags = dict()
            for ntag in range(len(tag_raw)):
                tags[ntag] = [tag_raw[ntag]]
            profile = correct_from_device_parameters(
                profile, None, feed_names, tags, lb, ub)
            memento['device_bounds'] = profile[profile != profile_old].index
            profile_old = profile.copy()
        if corrections['outliers']:
            if (('ENERGY' in profile.name)
                    & ('SELF_READING' not in profile.name)):
                profile = fill_outliers(
                    profile.to_frame().rename(columns={profile.name: 'real'}),
                    n_hours=3)
            memento['outliers'] = profile[profile != profile_old].index
            profile_old = profile.copy()
        if corrections['timestamps']:
            # add missing timestamps
            year = profile.index.year[min(20000, len(profile) - 1)]
            start_date = dt.combine(date(year, 1, 1), time(0, 0))
            end_date = dt.combine(date(year, 12, 31), time(23, 59, 50))
            profile = harmonize_timestamp(profile, start_date, end_date)
            memento['timestamps'] = profile.dropna().index.difference(
                profile_old.index)
            memento['missing'] = profile[profile.isna()].index.difference(
                profile_old.index)
        return profile, memento
        

    def aggregate_temporal(self, folder, corrections,
                           time_res_list=['10s', '1min', '15min', '60min']):
        '''
        Restructures the original datafiles into datafiles for publication.
        Creates resampled files for different temporal resolutions.

        Parameters
        ----------
        folder : str
            The folder containing the original datafiles
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')
        time_res_list : list
            List of strings containing the temporal resolutions

        Returns
        -------

        '''
        
        folder_res = os.path.join(folder, 'resampled')
        Path(folder_res).mkdir(exist_ok=True, parents=True)
        for file in os.listdir(folder_res):
            if file.endswith('.hdf5'):
                os.remove(os.path.join(folder_res, file))
        feed_replace = {
            'REACTIVE_POWER': 'Q',
            'REACTIVE_ENERGY': 'QE',
            'APPARENT_POWER': 'S',
            'APPARENT_ENERGY': 'SE',
            'POWER_FACTOR': 'PF',
            'POWER': 'P',
            'ENERGY': 'E',
            'CURRENT': 'I',
            'VOLTAGE': 'U',
            'L1': '1',
            'L2': '2',
            'L3': '3',
            'TOTAL': 'TOT',
            '_N': '',
            'ELECTRICITY_': '',
            '/INVERTER': 'INVERTER',
            '/IN': '',
            '/OUT': '',
            '_': '/'
            }
        obj_replace = {
            '.hdf5': '',
            'SINGLEFAMILYHOUSE_': 'SFH',
            'ELECTRICAL_SUBSTATION_': 'ES',
            'HEAT_STATION_': 'HS',
            'PV_POWER_PLANT_': 'PV',
            'WEATHER_STATION_': 'WS'}
        name_exclude = ['ERROR', 'FREQUENCY', 'CURRENT_N']
        # create a file to store data adjustments
        memento_filename = os.path.join(folder, 'resampled', 'memento.hdf5')
        memento_file = h5py.File(memento_filename, 'a')
        memento_store = pd.HDFStore(memento_filename)
        # temporal aggregation
        for source_file in os.listdir(folder):
            if source_file.endswith('.hdf5'):
                print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Adding file '
                      f'{source_file} to resampled data.')
                source_file_path = os.path.join(folder, source_file)
                # replace substrings in object name
                for key, value in obj_replace.items():
                    source_file = source_file.replace(key, value)
                file = h5py.File(source_file_path, 'r')
                visitor = H5ls()
                file.visititems(visitor)
                dset_names = visitor.names
                for dset_name_orig in dset_names:
                    dset_name = dset_name_orig.replace('/table', '')
                    print(dt.now().strftime("%m/%d/%Y, %H:%M:%S \t")
                          + dset_name)
                    if ('ELECTRICITY' not in dset_name
                            and source_file != 'WEATHER_ISFH'):
                        continue
                    # feeds containing substrings from name_exclude are
                    # feeds that will not be published
                    if any(ext in dset_name for ext in name_exclude):
                        continue
                    # import/export are energy feeds with different structure
                    if any(ext in dset_name for ext in ['IMPORT', 'EXPORT']):
                        split_len = 3
                        target_prepend = 'energy_'
                    else:
                        split_len = 2
                        target_prepend = ''
                    metadata = file[dset_name].attrs
                    profile = pd.read_hdf(source_file_path, dset_name)
                    # correct the data and write the corrected indices to
                    # memento
                    profile, memento = self.adjust_raw_data(
                        profile, corrections, metadata, dset_name)
                    for key, value in memento.items():
                        dset_name_memento = (source_file + '/' + dset_name
                                             + '/' + key)
                        memento_store.put(
                            dset_name_memento, pd.Series(value), append=False,
                            format='table', index=False, data_columns=True)
                    # replace substrings in feed name and prepend obj name
                    for key, value in feed_replace.items():
                        dset_name = dset_name.replace(key, value)
                    dset_name = source_file + '/' + dset_name
                    # the last three nodes of the dataset are always in/out,
                    # P/Q/.., and 1/2/.. Concat them to columns instead
                    dset_name, col_name = rreplace(
                        dset_name, '/', '_', split_len)
                    # prepend a new level to the dset_name depending on the
                    # object having PV panels installed
                    if any(obj in dset_name for obj in [
                            'ES1', 'PV1', 'WEATHER_ISFH']):
                        dset_name = 'MISC/' + dset_name
                    elif ('SFH' in dset_name and dset_name != 'WEATHER_ISFH'):
                        obj_nr = int(dset_name.split('/')[0][3:])
                        if obj_nr in self.pv_objs:
                            dset_name = 'WITH_PV/' + dset_name
                            # calculate estimated PV profile for houses with PV
                            if (col_name == 'P_TOT'
                                and 'HOUSEHOLD' in dset_name):
                                profile = self.adjust_profile_to_pv_infeed(
                                    folder, self.pv_objs[obj_nr], profile)
                        else:
                            dset_name = 'NO_PV/' + dset_name
                    for time_res in time_res_list:
                        # resample profile
                        profile_res = profile.resample(time_res).mean()
                        profile_res.name = col_name
                        profile_res.index = (
                            (profile_res.index - pd.Timestamp("1970-01-01"))
                            // pd.Timedelta('1s')
                        )
                        # create file
                        target_filename = os.path.join(
                            folder, 'resampled', target_prepend + 'data_'
                            + time_res + '.hdf5'
                        )
                        # we need to append columns which pytables does not
                        # support natively. Therefore, read the existing data,
                        # append a column and write it back
                        try:
                            old = pd.read_hdf(target_filename, dset_name)
                            profile_res = pd.concat([old, profile_res], axis=1)
                        # in the beginning, the file might not exists or the
                        # node might not exist
                        except (FileNotFoundError, KeyError, TypeError):
                            pass
                        target_file = h5py.File(target_filename, 'a')
                        store = pd.HDFStore(target_filename)
                        store.put(dset_name, profile_res, append=False,
                                  format='table', index=False,
                                  data_columns=True)
                        # append old metadata if key not already existent
                        # in new file
                        for key, value in metadata.items():
                            if key not in (target_file[dset_name].attrs
                                           .keys()):
                                target_file[dset_name].attrs.create(
                                    name=key, data=value)
                        target_file.close()
                        store.close()
                file.close()
        memento_store.close()
        memento_file.close()

    def aggregate_spatial(self, folder):
        '''
        Restructures the original datafiles into datafiles for publication.
        Creates a resampled file with different temporal resolution for the
        whole quarter.

        Parameters
        ----------
        folder : str
            The folder containing the original datafiles

        Returns
        -------

        '''
        
        folder_res = os.path.join(folder, 'resampled')
        for file in os.listdir(folder_res):
            if file == 'data_spatial.hdf5':
                os.remove(os.path.join(folder_res, file))
        # spatial aggregation
        name_exclude = ['HS1', 'PV1', 'WS1', 'WEATHER_ISFH']
        target_filename = os.path.join(
            folder_res, 'data_spatial.hdf5')
        included = dict(
            WITH_PV=dict(
                HOUSEHOLD={
                    '10s': [],
                    '1min': [],
                    '15min': [],
                    '60min': []
                },
                HEATPUMP={
                    '10s': [],
                    '1min': [],
                    '15min': [],
                    '60min': []
                },
            ),
            NO_PV=dict(
                HOUSEHOLD={
                    '10s': [],
                    '1min': [],
                    '15min': [],
                    '60min': []
                },
                HEATPUMP={
                    '10s': [],
                    '1min': [],
                    '15min': [],
                    '60min': []
                },
            ),
        )
        for source_file in os.listdir(folder_res):
            if not (source_file.startswith('data')
                    and source_file.endswith('hdf5')):
                continue
            temp_res = source_file.split('_')[1].replace('.hdf5', '')
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Add temporal '
                  f'resolution {temp_res} to spatial aggregation.')
            source_file_path = os.path.join(folder_res, source_file)
            if source_file_path == target_filename:
                continue
            file = h5py.File(source_file_path, 'r')
            visitor = H5ls()
            file.visititems(visitor)
            dset_names = visitor.names
            for dset_name_orig in dset_names:
                dset_name_orig = dset_name_orig.replace('/table', '')
                if any(ext in dset_name_orig for ext in name_exclude):
                        continue
                print(dt.now().strftime('%m/%d/%Y, %H:%M:%S')
                     + f'\t Dataset {dset_name_orig}.')
                profile = pd.read_hdf(source_file_path, dset_name_orig)
                profile.index = pd.to_datetime(profile.index, unit='s')
                cols = profile.columns.intersection(
                    ['P_TOT', 'Q_TOT', 'P_TOT_WITH_PV'])
                profile = profile[cols]
                # exclude profiles with missing (=NaN) values from spatial
                # aggregation. For the year 2018, exclude January to May from
                # this exclusion
                if '2018' in folder:
                    if profile.loc[profile.index.month > 5].isna().any().any():
                        print(dt.now().strftime('%m/%d/%Y, %H:%M:%S')
                              + f'\t Excluded dataset {dset_name_orig} from '
                              + 'spatial aggregation, because it contains NaNs.')
                        continue
                else:
                    if profile.isna().any().any():
                        print(dt.now().strftime('%m/%d/%Y, %H:%M:%S')
                              + f'\t Excluded dataset {dset_name_orig} from '
                              + 'spatial aggregation, because it contains NaNs.')
                        continue
                if 'ES1' in dset_name_orig:
                    dset_name_new = 'SUBSTATION' + '/' + temp_res
                else:
                    dset_name_new = (temp_res + '/'
                                     + dset_name_orig.split('/')[-1])
                    if 'NO_PV' in dset_name_orig:
                        dset_name_new = 'NO_PV' + '/' + dset_name_new
                    elif 'WITH_PV' in dset_name_orig:
                        dset_name_new = 'WITH_PV' + '/' + dset_name_new
                # save what profiles are included in each aggregation
                group, obj, feed = dset_name_orig.split('/')
                if group in included.keys():
                    included[group][feed][temp_res].append(
                        dset_name_orig.split('/')[1])
                # we need to append columns which pytables does not
                # support natively. Therefore, read the existing data,
                # append a column and write it back
                try:
                    old = pd.read_hdf(target_filename, dset_name_new)
                    old.index = pd.to_datetime(old.index, unit='s')
                    profile = old + profile
                # in the beginning, the file might not exists or the
                # node might not exist
                except (FileNotFoundError, KeyError):
                    pass
                store = pd.HDFStore(target_filename)
                store.put(dset_name_new, profile, append=False,
                          format='table', index=False,
                          data_columns=True)
                store.close()
        # attach included objects to the files' metadata
        file = h5py.File(target_filename, 'a')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            try:
                group, res, feed, table = dset_name.split('/')
            except ValueError:
                continue
            if not group in included.keys():
                continue
            file[dset_name].attrs.create(
                name='objects_included', data=included[group][feed][res])
        file.close()

    def prove_consistency(self, folder, corrections):
        '''
        Validates the consistency of the measurement data.

        Parameters
        ----------
        folder : str
            The folder containing the original datafiles
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')

        Returns
        -------

        '''
        
        validation = dict(ts_abs=dict(), hs_abs=dict(), ma=dict())
        val_dir = os.path.join(folder, 'validation')
        Path(val_dir).mkdir(exist_ok=True)
        source_file = os.path.join(
            folder, 'resampled', f'data_10s.hdf5')
        source_file_energy = os.path.join(
            folder, 'resampled', f'energy_data_10s.hdf5')
        file = h5py.File(source_file, 'r')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            if not any(substring in dset_name for substring in
                       ['HOUSEHOLD', 'HEATPUMP', 'TRANSFORMER']):
                continue
            dset_name = dset_name.replace('/table', '')
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Validating '
                  f'dataset {dset_name}.')
            obj = dset_name.split('/')[1]
            feed = dset_name.split('/')[2]
            profile = pd.read_hdf(source_file, dset_name)
            profile.index = pd.to_datetime(profile.index, unit='s')
            if 'P_TOT_WITH_PV' in profile.columns:
                profile = profile.drop('P_TOT', axis=1)
                profile = profile.rename(columns={'P_TOT_WITH_PV': 'P_TOT'})
            profile.columns = profile.columns.str.split('_', expand=True)
            profile_energy = pd.read_hdf(source_file_energy, dset_name)
            profile_energy.index = pd.to_datetime(profile_energy.index, unit='s')
            cols = profile_energy.columns.str.split('_', expand=True)
            profile_energy.columns = pd.MultiIndex.from_tuples(
                [('P' if col[0][0] == 'E' else col[0][0], col[1], col[2])
                 for col in cols])
            profile_energy = profile_energy.xs('IMPORT', axis=1, level=2)
            validation = self.concat_measurement_streams(
                    validation, profile, profile_energy, obj, feed)
                
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Finishing '
              'validation.. ')
        # absolute timeseries 10s values for P and U * I * PF
        validation['ts_abs'] = pd.concat(
            validation['ts_abs'].values(), keys=validation['ts_abs'].keys(),
            axis=1
        )
        self.compare_timeseries(validation['ts_abs'], strfile=val_dir)

        # absolute hourly sums for power and energy
        validation['hs_abs'] = pd.concat(
            validation['hs_abs'], keys=validation['hs_abs'].keys(), axis=1)
            # add multiindex level from split on underscores
        temp = (validation['hs_abs'].columns.to_frame().iloc[:, 0].str
                .split('_'))
        validation['hs_abs'].columns = pd.MultiIndex.from_arrays(
            [validation['hs_abs'].columns.get_level_values(1),
             temp.str[0], temp.str[1],
             validation['hs_abs'].columns.get_level_values(2),
             validation['hs_abs'].columns.get_level_values(3)],
            names=['measure', 'obj', 'feed', 'current', 'phase'])
        self.compare_aggregated_values(validation['hs_abs'], strfile=val_dir)
        
        # missing months to excel
        pd.Series(validation['ma']).reset_index().to_excel(
            os.path.join(val_dir, 'validation_months_available.xlsx'),
            index=False, header=['object', 'missing'])
        # yearly consumption from hourly consumption to excel
        (validation['hs_abs'].xs('Power', level=0, axis=1)
         .xs('P', axis=1, level=2).xs('TOT', axis=1, level=2)
         .sum().reset_index().to_excel(
             os.path.join(val_dir, 'ys_abs_per_node.xlsx'), index=False))

    def adjust_profile_to_pv_infeed(self, folder, installed_capacity, profile):
        '''
        Gets the PV profile of a single family house by using the ISFH profile
        and rescales it to the installed capacity of the house. Then adds the
        PV profile to the original profile to estimate the household load
        without PV feedin.

        Parameters
        ----------
        folder : str
            The folder containing the original datafiles
        installed_capacity : float
            The installed capacity of the PV panels of the house
        profile : pd.DataFrame
            The profile containing the load wit PV feedin

        Returns
        -------
        profile : pd.DataFrame
            The adjusted profile

        '''
        filename = os.path.join(folder, 'resampled', 'data_10s.hdf5')
        dset_name = 'MISC/PV1/PV/INVERTER/SOUTH'
        profile_pv = pd.read_hdf(filename, dset_name)['P_TOT']
        profile_pv.index = pd.to_datetime(profile_pv.index, unit='s')
        # the panels at the ISFH have 14.5 kWp. Linearly adjust this to the
        # actual installed capacity of the house
        profile_pv = profile_pv / 14.5 * installed_capacity
        profile = pd.DataFrame(profile)
        profile['P_TOT_WITH_PV'] = profile['POWER:TOTAL']
        profile['P_TOT'] = profile['P_TOT_WITH_PV'] + profile_pv
        profile = profile[['P_TOT_WITH_PV', 'P_TOT']]
        return profile

    def concat_measurement_streams(
            self, validation, profile, profile_energy, obj, feed):
        '''
        Extracts the relevant measurements like hourly data and the original
        data in 10s steps and adds them to the validation dictionary.

        Parameters
        ----------
        validation : dict
            Dictionary containing the processed data for annual and timeseries
            validation of all objects
        profile : pd.DataFrame
            Dataframe containg the original power data to be validated for the
            current object
        profile_energy : pd.DataFrame
            Dataframe containing the energy data
        obj : str
            The object corresponding to the profile
        feed : str
            The feed corresponding to the profile

        Returns
        -------
        validation : dict
            The validation dictionary extended by the data of the current
            object

        '''

        def build_imports(profile):
            '''
            Calculates imports from the overall profile. For apparent power,
            this is done via the power factor. Active power containts a
            negative sign.

            Parameters
            ----------
            profile : pd.DataFrame
                Dataframe containing the original data

            Returns
            -------
            profile : pd.DataFrame
                The original data with apparent power having an adjusted sign
            profile_import : pd.DataFrame
                A Dataframe containing only the exported power

            '''
            # split apparent power into positive and negative parts
            pf = profile.xs('PF', axis=1, level=0)
            pf[pf < 0] = -1
            pf[pf > 0] = 1
            apparent = profile.xs('S', axis=1, level=0).mul(pf)
            profile.loc[:, idx['S', apparent.columns]] = apparent.to_numpy()
            # build import
            profile_import = profile[profile > 0].fillna(0)
            return profile, profile_import

        res = 'D'
        # factor to go from power to energy
        if res == 'D':
            factor = 24
        elif res == 'H':
            factor = 1
        # absolute hourly sums for power and energy
        profile, profile_import = build_imports(profile)
        profile_energy = profile_energy.diff().fillna(0)
        profile_energy = profile_energy.dropna(axis=0, how='all')
        power = (profile_import[profile_energy.columns].resample(res).mean()
                 / 1e3 * factor)
        energy = profile_energy.resample(res).sum()
        validation['hs_abs'][obj + '_' + feed] = pd.concat(
            [power, energy], keys=['Power', 'Energy'], axis=1
            )

        # absolute timeseries 10s values for P and U * I * PF
        if feed != 'HEATPUMP':
            active = profile.xs('U', axis=1, level=0).multiply(
                profile.xs('I', axis=1, level=0), axis=1).multiply(
                    profile.xs('PF', axis=1, level=0), axis=1)
            active['TOT'] = active.sum(axis=1)
            validation['ts_abs'][obj + '_' + feed] = pd.concat(
                [profile.xs('P', axis=1, level=0).abs(), active],
                keys=['P', 'U * I * PF'], axis=1
                )
            # months available
            months = set(range(1, 13))
            missing = list(months.difference(profile.dropna(how='all').index
                                             .month.unique()))
            validation['ma'][obj] = missing
        return validation

    def detect_heating_rod_operation(self, folder, strfile=None):
        '''
        Detects three operation modes (heat pump, heating rod and pumps only)
        from the heat pump load curve.

        Parameters
        ----------
        folder : str
            The home folder containing both the validation and resampled
            folder.
        strfile : str
            Plots the data if a path to save a plot is given.
            The default is None.

        Returns
        -------
        None.

        '''

        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Detecting heat pump '
              'operation modes.')
        total = pd.DataFrame(index=['obj', 'method', 'Heat Pump',
                                      'Heating Rod', 'Pumps Only'])
        folder_res = os.path.join(folder, 'resampled')
        folder_val = os.path.join(folder, 'validation')
        filename = os.path.join(folder_res, 'data_10s.hdf5')
        file = h5py.File(filename, 'r')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            if not 'HEATPUMP' in dset_name:
                continue
            obj = dset_name.split('/')[1]
            for method in ['larger 4kW', 'Power Factor']:
                profile = pd.read_hdf(filename, dset_name).set_index('index')
                profile.index = pd.to_datetime(profile.index, unit='s')
                # assumption that a heat pump consumption larger than 4 kW
                # means heating rod operation
                if method == 'larger 4kW':
                    profile = profile[['P_TOT']]
                    profile.loc[profile['P_TOT'] < 4000, 'operation_mode'] = \
                        'Heat Pump'
                # assumption that the heating rod is an ohmic resistor, meaning
                # that apparent and active power are equal if the heat pump
                # runs in heating rod operation mode
                elif method == 'Power Factor':
                    profile = profile[['P_TOT', 'Q_TOT']]
                    profile.loc[
                        (profile['P_TOT'] > 100) & (profile['Q_TOT'] > 100),
                        'operation_mode'] = 'Heat Pump'
                # assumption that consumption < 100 W is pumps only
                profile.loc[profile['P_TOT'] < 100, 'operation_mode'] = \
                    'Pumps Only'
                profile['operation_mode'].fillna('Heating Rod', inplace=True)
                profile = profile.groupby('operation_mode').sum()
                profile.loc['method'] = method
                profile.loc['obj'] = obj
                total = pd.concat([total, profile['P_TOT']], axis=1)
        total = total.T.reset_index(drop=True)
        total.to_excel(
            os.path.join(folder_val, 'heat_pump_operation.xlsx'), index=False)
        if strfile:
            total = total.set_index(['obj', 'method']).sort_index()
            colors = plt.cm.Paired.colors
            total = total.unstack(level=-1)
            total = total / 360 / 1e3
            fig, ax = plt.subplots(figsize=(21, 9))
            (total['Heat Pump'] + total['Heating Rod']
             + total['Pumps Only']).plot(
                 kind='bar', color=[colors[1], colors[0]], rot=0, ax=ax)
            (total['Heating Rod'] + total['Pumps Only']).plot(
                kind='bar', color=[colors[3], colors[2]], rot=0, ax=ax)
            total['Pumps Only'].plot(
                kind='bar', color=[colors[5], colors[4]], rot=0, ax=ax)
            legend_labels = [f'{sink} ({method})' for sink, method
                             in total.columns]
            ax.legend(legend_labels)
            ax.set_ylabel('Annual energy in kWh')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=90)
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, 'heat_pump_operation.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()

    def compare_timeseries(self, profiles, strfile=None):
        '''
        Validates the measurements of active power vs. current * voltage
        * power factor in 10s resolution. Plots their data quality as a hexbin
        plot and saves percentiles to excel files.

        Parameters
        ----------
        profiles : pd.DataFrame
            A dataframe containing both measurement streams.
        strfile : str, optional
            The directory or filename of the plot. If it is a filename,
            filename is overwritten by strfile. If it is a directory, the plot
            is stored in this directory. If it is None, the plot is shown and
            not saved. The default is None.

        Returns
        -------

        '''
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
              + ' Plotting timeseries overview.')
        fig, ax = plt.subplots(figsize=(12, 9))
        # timesteps with nans are irrelevant for the plot, but we can't drop
        # the whole line. Fill them with a negative value and set axis limits
        # to 0 later
        profiles = profiles.fillna(-1)
        profiles = profiles.drop('ES1_TRANSFORMER', axis=1, level=0,
                                 errors='ignore')
        power = profiles.xs('P', axis=1, level=1)
        x = power.to_numpy().ravel()
        y = profiles.xs('U * I * PF', axis=1, level=1).to_numpy().ravel()
        pos = np.where((x > 0) & (y > 0))
        # calculate percentiles
        nonzeros = np.where(y > 0)
        perc_steps = [1, 5, 25, 50, 75, 95, 99]
        perc = np.percentile(x[nonzeros] / y[nonzeros], perc_steps)
        pd.Series(perc, index=perc_steps).to_excel(
            os.path.join(strfile, 'ts_rel_percentiles.xlsx'))
        # plot
        xmin = max(x[pos].min(), 1)
        xmax = x[pos].max()
        ymin = max(y[pos].min(), 1)
        ymax = y[pos].max()
        hb = ax.hexbin(x[pos], y[pos], gridsize=100, cmap='inferno',
                       bins='log', xscale='log', yscale='log')
        ax.axis([xmin, xmax, ymin, ymax])
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Counts')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3')
        ax.set_ylabel('Measured U * I * PF in W')
        ax.set_xlabel('Measured P in W')
        # percentiles of actual load to excel
        cons = power.xs('TOT', axis=1, level=1).to_numpy().ravel()
        perc_df = pd.Series(np.percentile(cons[cons >= 0], perc_steps),
                            index=perc_steps)
        perc_df.to_excel(os.path.join(strfile, 'ts_abs_PTOT_percentiles.xlsx'))
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, 'OVERVIEW_TS.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


    def compare_aggregated_values(self, profiles, strfile=None):
        '''
        Validates the measurements of power vs. energetic measurements in
        hourly resolution. Plots their data quality as a scatter plot and
        saves percentiles to excel files.

        Parameters
        ----------
        profiles : pd.DataFrame
            A dataframe containing both measurement streams.
        strfile : str, optional
            The directory or filename of the plot. If it is a filename,
            filename is overwritten by strfile. If it is a directory, the plot
            is stored in this directory. If it is None, the plot is shown and
            not saved. The default is None.

        Returns
        -------

        '''
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
              + ' Plotting P vs. E overview.')
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 17))
        y_fig = 0.68
        plt.subplots_adjust(bottom=1 - y_fig, top=0.97, right=0.95, left=0.07)
        # timesteps with nans are irrelevant for the plot, but we can't drop
        # the whole line. Fill them with a negative value and set axis limits
        # to 0 later
        profiles = profiles.fillna(-1)
        feed_dict = {
            'HOUSEHOLD': 0,
            'HEATPUMP': 1,
            'TRANSFORMER': 2
            }
        current_dict = {
            'S': 0,
            'P': 1,
            'Q': 2
            }
        objs = profiles.columns.get_level_values(1).unique()
        colors = cm.brg(np.linspace(0, 1, len(objs)))
        perc_steps = [1, 5, 25, 75, 95, 99]
        perc_df = pd.DataFrame(
            index=perc_steps, columns=pd.MultiIndex.from_product(
                [objs.append(pd.Index(['All'])),
                 profiles.columns.get_level_values(2).unique(),
                 profiles.columns.get_level_values(3).unique()]))
        # object specific
        for obj, c in zip(objs, colors):
            df = profiles.xs(obj, level=1, axis=1).copy()
            df = df.droplevel(3, axis=1)
            for feed in df.columns.get_level_values(1).unique():
                axis = axes[feed_dict[feed]]
                dff = df.xs(feed, axis=1, level=1)
                for current in dff.columns.get_level_values(1).unique():
                    ax = axis[current_dict[current]]
                    # calculate percentiles
                    x = dff.loc[:, idx['Power', current]].to_numpy().ravel()
                    y = dff.loc[:, idx['Energy', current]].to_numpy().ravel()
                    nonzeros = np.where(y > 0)[0]
                    try:
                        perc = np.percentile(x[nonzeros] / y[nonzeros],
                                             perc_steps)
                        perc_df.loc[perc_steps, idx[obj, feed, current]] = \
                            perc
                    except IndexError:
                        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
                              + f'\tValues for object {obj}, feed {feed} and '
                              f'current {current} are empty or zero. '
                              'Skipping..')
                    # plot
                    ax.scatter(x=dff.loc[:, idx['Power', current]],
                               y=dff.loc[:, idx['Energy', current]],
                               marker='x', color=c, label=obj)
        
        # get the overall percentiles per feed and current
        for feed, current in profiles.columns.droplevel(
                ['measure', 'obj', 'phase']).unique():
            x = (profiles.loc[:, idx['Power', :, feed, current, :]]
                 .to_numpy().ravel())
            y = (profiles.loc[:, idx['Energy', :, feed, current, :]]
                 .to_numpy().ravel())
            nonzeros = np.where(y > 0)[0]
            perc = np.percentile(x[nonzeros] / y[nonzeros], perc_steps)
            perc_df.loc[perc_steps, idx['All', feed, current]] = perc
        # global legend
        handles_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
                  if l not in labels[:i]]
        fig.legend(*zip(*unique), ncol=5, loc='upper left',
                   bbox_to_anchor=(0.07, 1 - y_fig - 0.07))
        # set y labels
        for feed in profiles.columns.get_level_values(2).unique():
            ax = axes[feed_dict[feed], 2]
            axt = ax.twinx()
            axt.set_ylabel(feed, rotation=90, labelpad=15)
            axt.yaxis.set_major_formatter(plt.NullFormatter())
            axt.yaxis.set_minor_locator(plt.NullLocator())
        for ax in axes.flatten():
            ax.tick_params(axis='y', rotation=45)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        # set column titles
        for current in profiles.columns.get_level_values(3).unique():
            ax = axes[0, current_dict[current]]
            ax.set_title(current)
        # set lower left point to (0, 0). Additionally, set upper ylim to the
        # upper xlim. Energy measurements sometimes produce errors themselves,
        # but we only want to look at power measurements. Therefore, let power
        # measurements be the leader concerning axis limits
        for ax in axes.flatten():
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=ax.get_xlim()[1])
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3')
        fig.text(0, 1 - y_fig / 2,
                 r'$\frac{E(t_2)-E(t_1)}{t_2-t_1}$ in kWh/h', va='center',
                 rotation='vertical', fontsize=22
        )
        fig.text(
            0.5, 1 - y_fig - 0.05,
            r'$\frac{1}{t_2-t_1} \int_{t_1}^{t_2} P(t) \,dt$ in kWh/h',
            ha='center', fontsize=22
        )
        perc_df.unstack().reset_index().to_excel(
            os.path.join(strfile, 'ys_rel_percentiles.xlsx'))
        # percentiles of actual hourly consumption
        consumption = (profiles.drop('ES1', level=1, axis=1)
                       .drop('Q', axis=1, level=3).xs('TOT', axis=1, level=4)
                       .xs('Power', level=0, axis=1)
        )
        perc_steps = [1, 5, 25, 50, 75, 95, 99]
        perc_df = pd.Series(np.percentile(consumption, perc_steps),
                            index=perc_steps)
        perc_df.to_excel(os.path.join(strfile, 'hs_abs_PandS_percentiles.xlsx'))
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, 'OVERVIEW_P_VS_E_SCATTER.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, dpi=300)
            plt.close()
        else:
            plt.show()
