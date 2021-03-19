'''
The implementation of this file is very close to the demandlib, with only some
code adjusted and added by ourselves. We would like to thank them for their
code under an open license and please visit their GitHub repository at
https://github.com/oemof/demandlib
'''

import calendar
import pandas as pd
import datetime
import os
from pathlib import Path

import pytz
import numpy as np


def add_weekdays2df(time_df, holidays=None, holiday_is_sunday=False):
    '''
    Giving back a DataFrame containing weekdays and optionally holidays for
     the given year.

    Parameters
    ----------
    time_df : pandas DataFrame
        DataFrame to which the weekdays should be added
    holidays : array with information for every hour of the year, if holiday or
        not (0: holiday, 1: no holiday)
    holiday_is_sunday : boolean
        If set to True, all holidays (0) will be set to sundays (7).

    Returns
    -------
    pandas.DataFrame : DataFrame with weekdays


    '''
    day_dict = {
        1: 'Werktag',
        2: 'Werktag',
        3: 'Werktag',
        4: 'Werktag',
        5: 'Werktag',
        6: 'Samstag',
        7: 'Sonntag'
        }
    time_df['weekday'] = time_df.index.weekday + 1
    time_df['date'] = time_df.index.date
    time_df['time'] = time_df.index.time

    # Set weekday to Holiday (0) for all holidays
    if holidays is not None:
        if isinstance(holidays, dict):
            holidays = list(holidays.keys())
        time_df['weekday'].mask(pd.to_datetime(time_df['date']).isin(
            pd.to_datetime(holidays)), 0, True)

    if holiday_is_sunday:
        time_df.weekday.mask(time_df.weekday == 0, 7, True)
    time_df.replace({'weekday': day_dict}, inplace=True)
    return time_df


class ElecSlp:
    '''
    Generate electrical standardized load profiles based on the BDEW method.

    Parameters
    ----------
    year : integer
        Year of the demand series.
    seasons : dictionary
        Describing the time ranges for summer, winter and transition periods.
    holidays : dictionary or list
        The keys of the dictionary or the items of the list should be datetime
        objects of the days that are holidays.
    '''

    def __init__(self, year, seasons=None, holidays=None):
        if calendar.isleap(year):
            hoy = 8784
        else:
            hoy = 8760
        self.date_time_index = pd.date_range(
            pd.datetime(year, 1, 1, 0), periods=hoy * 4, freq='15Min',
            tz='Europe/Berlin')
        if seasons is None:
            self.seasons = {
                'summer1': [5, 15, 9, 14],  # summer: 15.05. to 14.09
                'transition1': [3, 21, 5, 14],  # transition1 :21.03. to 14.05
                'transition2': [9, 15, 10, 31],  # transition2 :15.09. to 31.10
                'winter1': [1, 1, 3, 20],  # winter1:  01.01. to 20.03
                'winter2': [11, 1, 12, 31],  # winter2: 01.11. to 31.12
            }
        else:
            self.seasons = seasons
        self.year = year
        self.slp_frame = self.all_load_profiles(self.date_time_index,
                                                holidays=holidays)

    def all_load_profiles(self, time_df, holidays=None):
        '''
        Calculate all load profiles for a given datetime

        Parameters
        ----------
        time_df : pd.DatetimeIndex
            The dates
        holidays : array with information for every hour of the year, if
            holiday or not (0: holiday, 1: no holiday)

        Returns
        -------
        new_df : pd.DataFrame
            The profiles

        '''
        new_df = pd.concat(
            [self.create_bdew_load_profiles(time_df, holidays=holidays),
             self.create_bdew_heat_pump_load_profile(
                 time_df, holidays=holidays)],
            axis=1
        )
        return new_df

    def create_bdew_load_profiles(self, dt_index, holidays=None, dynamic=True):
        '''
        Calculates the hourly electricity standard load profile of the BDEW

        Parameters
        ----------
        dt_index : pd.DatetimeIndex
            The dates
        holidays : array with information for every hour of the year, if
            holiday or not (0: holiday, 1: no holiday)
        dynamic : bool, optional
            Return the dynamic or static SLP

        Returns
        -------
        new_df : pd.DataFrame
            The profile
        '''

        # define file path of slp csv data
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'slpe.csv')

        # Read standard load profile series from csv file
        selp_series = pd.read_csv(file_path, delimiter=';', index_col=0,
                                  header=[0, 1])
        selp_series = selp_series.unstack().reset_index()
        selp_series.columns = ['season', 'weekday', 'time', 'h0']
        selp_series['time'] = pd.to_datetime(
            selp_series['time'], format='%H:%M').dt.time

        # Create empty DataFrame to take the results.
        new_df = pd.DataFrame(index=dt_index)
        new_df = add_weekdays2df(new_df, holidays=holidays,
                                 holiday_is_sunday=True)
        
        for p in self.seasons.keys():
            a = pytz.timezone('Europe/Berlin').localize(
                datetime.datetime(self.year, self.seasons[p][0],
                                  self.seasons[p][1], 0, 0))
            b = pytz.timezone('Europe/Berlin').localize(
                datetime.datetime(self.year, self.seasons[p][2],
                                  self.seasons[p][3], 23, 59))
            new_df.loc[a:b, 'season'] = p[:-1]
        new_df = (new_df.reset_index().merge(
            selp_series, on=['weekday', 'season', 'time'])
            .set_index('index').sort_index())
        new_df.drop('date', axis=1, inplace=True)
        if dynamic:
            new_df['h0'] = self.dynamic_h0(new_df)
        new_df.drop(['weekday', 'season', 'time'], axis=1, inplace=True)
        return new_df.div(new_df.sum(axis=0))

    def create_bdew_heat_pump_load_profile(self, dt_index, holidays=None):
        '''
        Calculates the hourly electricity heat pump standard load profile of
        the BDEW

        Parameters
        ----------
        dt_index : pd.DatetimeIndex
            The dates
        holidays : array with information for every hour of the year, if
            holiday or not (0: holiday, 1: no holiday)

        Returns
        -------
        new_df : pd.DataFrame
            The profile
        '''
        # get fixed values from slp
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'slphp.csv')
        selp_series = pd.read_csv(file_path, delimiter=';', index_col=0,
                                  header=0)
        selp_series.index = pd.to_datetime(
            [index[0] for index in selp_series.index.str.split('-')],
            format='%H:%M').time
        selp_series.columns = (selp_series.columns.str.replace('<', '')
                               .str.replace('=', '').str.replace('>', ''))
        selp_series = (selp_series.stack().reset_index()
                       .rename(columns={'level_0': 'hour', 'level_1': 'tmd',
                                        0: 'value'})
        )
        selp_series['tmd'] = selp_series['tmd'].astype(float)
        # get temperatures
        file_path = os.path.join(
            Path(os.path.dirname(__file__)).parent, f'Ohrberg_{self.year}',
            'WEATHER_STATION_1.hdf5')
        dset_name = 'WEATHER_SERVICE/IN/WEATHER_TEMPERATURE_TOTAL'
        weather = pd.read_hdf(file_path, dset_name)
        weather = weather.resample('15min').mean()
        weather.index = weather.index.tz_localize(
            'Europe/Berlin', nonexistent='shift_forward', ambiguous='NaT')
        weather = weather.groupby(weather.index).first().reindex(
            dt_index, method='ffill')
        weather = pd.DataFrame(weather)
        # https://www.bdew.de/media/documents/LPuVe-Praxisleitfaden.pdf
        tmd = weather[np.isin(weather.index.time,
                        [datetime.time(7, 0), datetime.time(14, 0),
                         datetime.time(20, 30)])]
        tmd.loc[tmd.index.hour == 7, 'weight'] = 0.25
        tmd.loc[tmd.index.hour == 14, 'weight'] = 0.25
        tmd.loc[tmd.index.hour == 20, 'weight'] = 0.5
        tmd = (tmd['TEMPERATURE:TOTAL'] * tmd['weight']).groupby(
            tmd.index.date).sum()
        tmd = (0.5 * tmd + 0.3 * np.roll(tmd, 1) + 0.15 * np.roll(tmd, 2)
               + 0.05 * np.roll(tmd, 3))
        tmd = tmd.apply(np.floor)
        tmd[tmd < -14] = -14
        tmd[tmd > 18] = 18
        tmd = tmd.reset_index()
        new_df = pd.DataFrame(index=dt_index)
        new_df = add_weekdays2df(new_df, holidays=holidays,
                                 holiday_is_sunday=True)
        new_df = (new_df.merge(tmd, left_on='date', right_on='index')
                  .rename(columns={0: 'tmd'}))
        new_df = new_df.merge(selp_series, left_on=['time', 'tmd'],
                              right_on=['hour', 'tmd'])
        new_df = pd.DataFrame(
            new_df['value'].to_numpy(),
            index=pd.to_datetime(new_df['date'].astype(str) + ' '
                                 + new_df['time'].astype(str))
        )
        new_df = new_df.sort_index()
        new_df.columns = ['BDEW HP']
        new_df.index = dt_index
        return new_df.div(new_df.sum())

    def dynamic_h0(self, new_df):
        '''
        Function to generate the dynamic SLP H0

        Parameters
        ----------
        new_df : pd.DataFrame
            The non-dynamic dataframe

        Returns
        -------
        new_df : pd.DataFrame
            The dynamic dataframe

        '''
        a4 = -3.92 * 1e-10
        a3 = 3.2 * 1e-7
        a2 = -7.02 * 1e-5
        a1 = 2.1 * 1e-3
        a0 = 1.24
        t = new_df.index.dayofyear
        fT = np.round(a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0, 4)
        return new_df['h0'] * fT

    def get_profile(self, ann_el_demand_per_sector):
        '''
        Get the profiles for the given annual demand

        Parameters
        ----------
        ann_el_demand_per_sector : dictionary
            Key: sector, value: annual value

        Returns
        -------
        pandas.DataFrame : Table with all profiles

        '''
        return self.slp_frame.multiply(pd.Series(
            ann_el_demand_per_sector), axis=1)
