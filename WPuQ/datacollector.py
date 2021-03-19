from time import sleep
from datetime import date
from datetime import datetime as dt
from datetime import time
from calendar import monthrange
from json.decoder import JSONDecodeError
from pathlib import Path
import requests
import warnings
import os

import pandas as pd
import h5py
import tables

from .quarters import quarter_names, heat_objs, pv_objs
from .util import (
    month_year_iter,
    get_devices,
    get_parameters_from_device,
    fill_outliers,
    correct_from_device_parameters,
    harmonize_timestamp)

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

idx = pd.IndexSlice

class Datacollector():
    '''
    This class downloads the data of the WPuQ project and saves it to HDF5
    files.
    
    Parameters
    ----------
    apikey : str: 
        The apikey

    Attributes
    ----------
    apikey : str
        Store of the apikey
    url_feeds : str
        The URL to get the feeds from
    devices : pandas.DataFrame
        Dataframe containing information about the devices used to measure the
        timeseries
    participants : pandas.DataFrame
        Dataframe containing information about the participating households
    '''
    def __init__(self, apikey):
        self.apikey = apikey
        self.url_feeds = 'http://monitoring.wpuq.de/feed/list.json?apikey='
        
        self.participants = pd.read_excel('participants.xlsx', header=0,
                                          usecols="E,G,H")

    def get_objects(self, quarter):
        '''
        Gets the objects from a given quarter

        Parameters
        ----------
        quarter : str
            The quarter to get objects from

        Raises
        ------
        NotImplementedError
            Unknown quarter.

        Returns
        -------
        None

        '''
        self.quarter = quarter
        url_users = ('https://monitoring.wpuq.de/userroles/getusersbyrole'
            'id?roleid=')
        if quarter not in quarter_names:
            raise NotImplementedError('Unknown quarter.')
        self.url = url_users + quarter_names[quarter] + '&apikey=' + self.apikey
        self.objects = requests.get(self.url).json()
        self.heat_objs = heat_objs[quarter]
        self.pv_objs = pv_objs[quarter]

    def get_data(self, start_month, end_month, time_interval,
                 corrections, weather_data=True):
        '''
        Downloads the data and processes it to HDF5 files

        Parameters
        ----------
        start_month : str
            The start month in the form 'mm-yyyy'
        end_month : str
            The end month in the form 'mm-yyyy'
        time_interval : int
            The time interval in seconds
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')
        weather_data : bool
            Decides if weather data should be downloaded

        Returns
        -------

        '''
        if (dt.strptime(start_month, "%m-%Y") > dt.now() or
            dt.strptime(end_month, "%m-%Y") > dt.now()):
            raise NotImplementedError('Start or enddate are in the future.')
        self.start_date = dt.strptime(start_month, "%m-%Y")
        self.end_date = dt.strptime(end_month, "%m-%Y")
        if self.start_date.year != self.end_date.year:
            raise NotImplementedError('Start and end month must be in the '
                                      'same year.')
        self.time_interval = time_interval
        
        folder = self.quarter + '_' + str(self.start_date.year)
        Path(folder).mkdir(exist_ok=True)
        if weather_data:
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Weather data.')
            self.get_weather_data(folder, self.start_date.year)
        for idx, obj in enumerate(self.objects):
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Object no. '
                  + str(idx) + ' of ' + str(len(self.objects)) + '.')
            part_idx = self.participants.loc[
                self.participants['Adresse'] == obj['location']].index
            url = self.url_feeds + obj['apikey_read']
            feeds = requests.get(url).json()
            name_raw = obj['username'].split(':')
            names = dict()
            for nname in range(len(name_raw)):
                names[nname] = name_raw[nname].split('.')
            # skip dummy objects
            if not 1 in names.keys():
                continue
            h5_filename = os.path.join(folder, '_'.join(names[1]) + '.hdf5')
            if os.path.isfile(h5_filename):
                print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' File '
                      f'{h5_filename} already exists. Skipping..')
                continue
            file = h5py.File(h5_filename, 'w')
            # dates
            for year, month in month_year_iter(
                    self.start_date.month, self.start_date.year,
                    self.end_date.month, self.end_date.year):
                print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Month '
                  + str(month) + ' in year ' + str(year) + '.')
                data = {}
                n_days = monthrange(year, month)[1]
                bulk_days = 10
                for idx_f, feed in enumerate(feeds):
                    print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
                          + ' Feed no. ' + str(idx_f + 1) + ' of '
                          + str(len(feeds)) + '.')
                    if (('DATALOGGER' in feed['tag'])
                            or ('ERROR' in feed['name'])):
                        continue
                    # data size is restricted by the SQl server. Therefore,
                    # split the query size into 10 days of data
                    profile = pd.DataFrame()
                    for day in range(1, n_days + 1, bulk_days):
                        start_epoch = int(
                            dt.strptime(f"{day}-{month}-{year} 00:00:00",
                                        "%d-%m-%Y %H:%M:%S").timestamp()) * 1e3
                        if day + bulk_days <= n_days:
                            end_epoch = int(dt.strptime(
                                f"{day+bulk_days-1}-{month}-{year} 23:59:59",
                                "%d-%m-%Y %H:%M:%S").timestamp()) * 1e3
                        else:
                            end_epoch = int(dt.strptime(
                                f"{n_days}-{month}-{year} 23:59:59",
                                "%d-%m-%Y %H:%M:%S").timestamp()) * 1e3
                        profile = profile.append(self.load_feed_data(
                            feed['id'], start_epoch, end_epoch,
                            obj['apikey_read']), ignore_index=True)
                    # tag names
                    tag_raw = feed['tag'].split(':')
                    tags = dict()
                    for ntag in range(len(tag_raw)):
                        tags[ntag] = tag_raw[ntag].split('.')
                    # feed names
                    feed_names = dict()
                    feed_names_raw = feed['name'].split(':')
                    for nfeed in range(len(feed_names_raw)):
                        feed_names[nfeed] = feed_names_raw[nfeed]
                    # proove id
                    object_nr, device_nr, mp_nr, tags[0] = self.proove_id(
                        names[1], tags)
                    if profile.empty:
                        err_margin = None
                        measure_device = None
                        measure_range = None
                        unit = None
                    if not profile.empty:
                        start_date = dt.combine(
                            date(year, month, 1), time(0, 0))
                        end_date = dt.combine(
                            date(year, month, n_days), time(23, 59, 50))
                        (profile, err_margin, measure_device,
                         measure_range, unit) = self.adjust_raw_data(
                             corrections, profile, obj, feed, tags, names,
                             feed_names, start_date, end_date)
                        dset_name = (tags[0][0] + '/' + tags[1][0] + '/'
                                     + tags[len(tags) - 1][0] + '_'
                                     + '_'.join(feed_names.values()))
                        store = pd.HDFStore(h5_filename)
                        if month == self.start_date.month:
                            append = False
                        else:
                            append = True
                        store.put(
                            dset_name, profile[feed['name']].astype(float),
                            append=append, format='table', index=False)
                        store.close()
                        data[dset_name] = profile[feed['name']]
                        metadata = {'error_margin': err_margin,
                                    'measure_device': measure_device,
                                    'measure_range': measure_range,
                                    'unit': unit}
                        # add object metadata to file
                        if not part_idx.empty:
                            metadata['living_space'] = self.participants.loc[
                                part_idx, 'Wohnfläche'].values[0]
                            metadata['n_inhabitants'] = self.participants.loc[
                                part_idx, 'Bewohner'].values[0]
                        for key, value in metadata.items():
                            if type(value) == str:
                                value = value.encode('utf8')
                            file[dset_name].attrs.create(name=key, data=value)
                        
                # correct pv data for first object
                if (not profile.empty and obj == self.objects[0]):
                    self.correct_pv_dc(data, corrections, h5_filename, append,
                                       start_date, end_date)
                if object_nr in self.heat_objs:
                    data = self.set_heat_energy(data, names[1][1], h5_filename,
                                                append=append)
            file.close()
                    
    def get_weather_data(self, folder, year):
        '''
        Creates an hdf5 file from measured ISFH weather data.

        Parameters
        ----------
        folder : str
            The folder to write the hdf5 file to
        year : int
            The year go get the data from

        Raises
        ------
        NotImplementedError
            Raised if weather data is requested for an unknown quarter

        Returns
        -------

        '''
        if self.quarter != 'Ohrberg':
            raise NotImplementedError(
                f'Weather data for {self.quarter} not available.')
        profile = pd.read_excel(f'weather_isfh_{year}.xlsx')
        units = profile.iloc[0]
        profile = profile.drop(index=0)
        profile.index = pd.to_datetime(profile['Date'].astype(str) + ' '
                                       + profile['Time'].astype(str))
        profile = profile.drop(columns=['Date', 'Time'])
        profile = profile.astype(float)
        profile = profile.drop(columns=['Tmodul'])
        start_date = dt.combine(date(year, 1, 1), time(0, 0))
        end_date = dt.combine(date(year, 12, 31), time(23, 59, 50))
        profile = harmonize_timestamp(profile, start_date, end_date)
        # save to hdf5
        h5_filename = os.path.join(folder, 'WEATHER_ISFH.hdf5')
        file = h5py.File(h5_filename, 'w')
        store = pd.HDFStore(h5_filename)
        for dset_name in profile.columns:
            store.put(dset_name, profile[dset_name], append=True,
                      format='table', index=False)
            file[dset_name].attrs.create(name='unit', data=units[dset_name])
        store.close()
        file.close()

    def load_feed_data(self, feed_id, start_epoch, end_epoch, apikey_read):
        '''
        Downloads the data of a given feed

        Parameters
        ----------
        feed_id : str
            The id of the feed
        start_epoch : int
            POSIX timestamp to start at
        end_epoch : int
            POSIX timestamp to end at
        apikey_read : str
            The apikey to read the feed

        Returns
        -------
        data : list
            The downloaded data.

        '''
        url_data = 'http://monitoring.wpuq.de/feed/data.json?id='
        url = (url_data + feed_id + '&start=' + str(start_epoch) + '&end=' +
               str(end_epoch) + '&interval=' + str(self.time_interval) +
               '&skipmissing=1&limitinterval=1&apikey=' + apikey_read)
        n_attempts = 1
        # the URL sometimes isn't available for a short time. Wait 30 seconds
        # if the request failed and try again
        while n_attempts < 180:
            try:
                data = requests.get(url).json()
                break
            except JSONDecodeError:
                print(f'Requesting URL failed in attempt {n_attempts}. ' + 
                      'Waiting 30 seconds until next attempt.')
                sleep(30)
            n_attempts += 1
        return data

    def adjust_raw_data(self, corrections, profile, obj, feed, tags, names,
                        feed_names, start_date, end_date):
        '''
        Corrects the raw data

        Parameters
        ----------
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')
        profile : pd.DataFrame
            The data to correct
        obj : dict
            Description of the object
        feed : dict
            Description of the feed
        tags : dict
            A processed dict of feed['tag']
        names : dict
            A processed dict of obj['username']
        feed_names : dict
            A processed dict of feed['name']
        start_date : datetime.datetime
            The desired start date
        end_date : datetime.datetime
            The desired end date

        Returns
        -------
        profile : pd.DataFrame
            The corrected data
        err_margin : str
            The error margin of the measurement device
        measure_device : str
            The name of the measurement device
        measure_range : str
            The measurement range of the measurement device
        unit : str
            The unit of the profile

        '''
        profile.columns = ['Timestamp', feed['name']]
        profile['Timestamp'] = pd.to_datetime(
            profile['Timestamp'], unit='ms')
        profile = profile.set_index('Timestamp')
        if obj == self.objects[0]:
            # correct pv-dc power for east-west alignment
            self.prepare_pv_dc_correction(feed, profile)
        devices = get_devices()
        lb, ub, err_margin, measure_range, measure_device, unit = \
            get_parameters_from_device(devices, names, tags, feed, feed_names)
        if corrections['device_bounds']:
            profile = correct_from_device_parameters(
                profile, feed['name'], feed_names, tags, lb, ub)

        datapoint_name = (tags[len(tags) - 1][0] + '_'
                          + '_'.join(feed_names.values()))
        # fill outliers. Don't apply it to Voltage, Current, frequency
        # power, solar irradiance and power factor, because filloutliers
        # detects false positives
        if corrections['outliers']:
            if (('ENERGY' in datapoint_name)
                    & ('SELF_READING' not in feed_names[1])):
                profile[feed['name']] = fill_outliers(profile, n_hours=3)
        if corrections['timestamps']:
            # add missing timestamps
            profile = harmonize_timestamp(profile, start_date, end_date)
        return profile, err_margin, measure_device, measure_range, unit

    def proove_id(self, name2, tags):
        '''
        Get the ids of a feed in an object

        Parameters
        ----------
        name2 : str
            A part of obj['username']
        tags : dict
            A processed dict of feed['tag']

        Returns
        -------
        object_nr : int
            The number of the object
        device_nr : int
            The number of the device.
        mp_nr : int
            A number of the feed
        tags[0] : list
            A part of tags

        '''
        if len(name2) > 1:
            # maybe try catch value error here, because object_nr is not a number?
            object_nr = int(name2[1])
            if not object_nr:
                object_nr = 1
        else:
            object_nr = 1

        if len(tags[0]) > 1:
            # device_nr can be a string. set to 1 then and concat the tags
            try:
                device_nr = int(tags[0][1])
            except ValueError:
                device_nr = 1
                tags[0][0] = tags[0][0] + '_' + tags[0][1]
        else:
            device_nr = 1

        if len(tags[1]) > 1:
            mp_nr = int(tags[1][1])
            if not mp_nr:
                mp_nr = 1
        else:
            mp_nr = 1
        return object_nr, device_nr, mp_nr, tags[0]

    def prepare_pv_dc_correction(self, feed, profile):
        '''
        The profile of east and west aligned PV powerplants has to be adjusted
        later. Create DataFrames that include necessary data

        Parameters
        ----------
        feed : dict
            Dictionary containing information about the feed
        profile : pandas.DataFrame
            The data of the feed

        Returns
        -------

        '''
        if 'EAST' in feed['tag']:
            if feed['name'] == 'POWER:DC_TOTAL':
                self.hp_east = profile.copy()
            elif feed['name'] in ('CURRENT:DC2', 'VOLTAGE:DC2'):
                self.hp_east = (profile.merge(
                    self.hp_east, on='Timestamp', how='outer').sort_index()
                    .fillna(0))
        elif 'WEST' in feed['tag']:
            if feed['name'] == 'VOLTAGE:DC2':
                self.hp_west = profile.copy()
            elif feed['name'] in ('CURRENT:DC2', 'POWER:DC_TOTAL'):
                self.hp_west = (profile.merge(
                    self.hp_west, on='Timestamp', how='outer').sort_index()
                    .fillna(0))

    def correct_pv_dc(self, data, corrections, h5_filename, append, start_date,
                      end_date):
        '''
        Corrects the PV data where the measured current and voltage suggests
        that the measured power data is wrong

        Parameters
        ----------
        data : dict
            Dictionary containing all Dataframes of the object
        corrections : dict
            Dictionary deciding how the raw data is corrected. Possible keys
            are ('timestamps', 'device_bounds', 'outliers')
        h5_filename : str
            Name of the HDF5 file of the object
        append : bool
            Specifies if the data is appended to the HDF5 file
        start_date : datetime.datetime
            Datetime object setting the intended start date
        end_date : datetime.datetime
            Datetime object setting the intended end date

        Returns
        -------

        '''
        for ori in ['east', 'west']:
            dset_name = 'PV_INVERTER_{}/OUT/ELECTRICITY_POWER_DC_TOTAL'.format(
                ori.upper())
            df = getattr(self, 'hp_' + ori)
            df = df.loc[data[dset_name].index]
            iv = df['CURRENT:DC2'] * df['VOLTAGE:DC2'] * 78 / 18.25
            p_dc_diff = df['POWER:DC_TOTAL'] - iv
            iv_idx = df.loc[p_dc_diff > (iv * 0.1)].index
            df.loc[iv_idx, 'POWER:DC_TOTAL'] = (
                df.loc[iv_idx, 'CURRENT:DC2']
                * df.loc[iv_idx, 'VOLTAGE:DC2']
                * 78 / 18.25)
            if corrections['outliers']:
                df['POWER:DC_TOTAL'] = fill_outliers(
                    pd.DataFrame(df['POWER:DC_TOTAL'].values, columns=['real'],
                                 index=df.index))
            if corrections['timestamps']:
                df = harmonize_timestamp(df, start_date, end_date)
            # set values to P_AC where P_DC < P_AC
            hp_idx = df.loc[df['POWER:DC_TOTAL'] < data[dset_name]].index
            df.loc[hp_idx, 'POWER:DC_TOTAL'] = data[dset_name].loc[hp_idx]
            setattr(self, 'hp_' + ori, df)
            store = pd.HDFStore(h5_filename)
            # preserve metadata
            metadata = store.get_node(dset_name)._v_attrs
            store.put(dset_name, df['POWER:DC_TOTAL'], format='table', 
                      append=append, index=False)
            store.get_storer(dset_name)._v_attrs = metadata
            store.close()

    def set_heat_energy(self, data, name, h5_filename, append):
        '''
        Adds extra information about heat energy to the HDF5 files

        Parameters
        ----------
        data : dict
            Dictionary containing all Dataframes of the object
        name : str
            A part of obj['username']
        h5_filename : str
            Name of the HDF5 file of the object
        append : bool
            Specifies if the data is appended to the HDF5 file

        Returns
        -------
        data : dict
            Dictionary containing the adjusted Dataframes

        '''
        dset_names = []
        if 'HEATPUMP/IN/HEAT_FLOW_RATE_TOTAL' in data:
            data['HEATPUMP/IN/HEAT_POWER_TOTAL_ber'] = (
                4.19 / 3600 * 1e6
                * data['HEATPUMP/IN/HEAT_FLOW_RATE_TOTAL']
                * (data['HEATPUMP/IN/HEAT_TEMPERATURE_RETURN']
                   - data['HEATPUMP/IN/HEAT_TEMPERATURE_FLOW']))
            data['HEATPUMP/IN/HEAT_ENERGY_TOTAL_ber'] = \
                data['HEATPUMP/IN/HEAT_POWER_TOTAL_ber'].cumsum()
            dset_names.extend(['HEATPUMP/IN/HEAT_POWER_TOTAL_ber',
                               'HEATPUMP/IN/HEAT_ENERGY_TOTAL_ber'])
        if 'HEATPUMP/OUT/HEAT_FLOW_RATE_TOTAL' in data:
            data['HEATPUMP/OUT/HEAT_POWER_TOTAL_ber'] = (
                4.19 / 3600 * 1e6
                * data['HEATPUMP/OUT/HEAT_FLOW_RATE_TOTAL']
                * (data['HEATPUMP/OUT/HEAT_TEMPERATURE_FLOW']
                   - data['HEATPUMP/OUT/HEAT_TEMPERATURE_RETURN']))
            data['HEATPUMP/OUT/HEAT_ENERGY_TOTAL_ber'] = \
                data['HEATPUMP/OUT/HEAT_POWER_TOTAL_ber'].cumsum()
            dset_names.extend(['HEATPUMP/OUT/HEAT_POWER_TOTAL_ber',
                               'HEATPUMP/OUT/HEAT_ENERGY_TOTAL_ber'])
        if 'HOT_WATER_TANK/IN/HEAT_FLOW_RATE_TOTAL' in data:
            data['HOT_WATER_TANK/IN/HEAT_POWER_TOTAL_ber'] = (
                4.19 / 3600 * 1e6
                * data['HOT_WATER_TANK/IN/HEAT_FLOW_RATE_TOTAL']
                * (data['HOT_WATER_TANK/IN/HEAT_TEMPERATURE_FLOW']
                   - data['HOT_WATER_TANK/IN/HEAT_TEMPERATURE_RETURN']))
            data['HOT_WATER_TANK/IN/HEAT_ENERGY_TOTAL_ber'] = \
                data['HOT_WATER_TANK/IN/HEAT_POWER_TOTAL_ber'].cumsum()
            dset_names.extend(['HOT_WATER_TANK/IN/HEAT_POWER_TOTAL_ber',
                               'HOT_WATER_TANK/IN/HEAT_ENERGY_TOTAL_ber'])
        if 'SOLAR_THERMAL_COLLECTOR/OUT/HEAT_FLOW_RATE_TOTAL' in data:
            data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_POWER_TOTAL_ber'] = (
                3.79 / 3600 * 1e6
                * data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_FLOW_RATE_TOTAL']
                * (data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_TEMPERATURE_FLOW']
                   - data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_TEMPERATURE_RETURN']))
            data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_ENERGY_TOTAL_ber'] = \
                data['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_POWER_TOTAL_ber'].cumsum()
            dset_names.extend(['SOLAR_THERMAL_COLLECTOR/OUT/HEAT_POWER_TOTAL_ber',
                               'SOLAR_THERMAL_COLLECTOR/OUT/HEAT_ENERGY_TOTAL_ber'])
        if name == 'HEAT_STATION':
            data['WASTE_HEAT_EXCHANGER/OUT/HEAT_POWER_ber'] = (
                4.19 / 3600 * 1e6
                * data['WASTE_HEAT_EXCHANGER/OUT/HEAT_FLOW_RATE_TOTAL']
                * (data['WASTE_HEAT_EXCHANGER/OUT/HEAT_TEMPERATURE_FLOW']
                   - data['WASTE_HEAT_EXCHANGER/OUT/HEAT_TEMPERATURE_RETURN']))
            data['WASTE_HEAT_EXCHANGER/OUT/HEAT_ENERGY_ber'] = \
                data['WASTE_HEAT_EXCHANGER/OUT/HEAT_POWER_ber'].cumsum()
            dset_names.extend(['WASTE_HEAT_EXCHANGER/OUT/HEAT_POWER_ber',
                               'WASTE_HEAT_EXCHANGER/OUT/HEAT_ENERGY_ber'])
        store = pd.HDFStore(h5_filename)
        for dset_name in dset_names:
            store.put(dset_name, data[dset_name], format='table',
                      append=append, index=False)
        store.close()
        return data
