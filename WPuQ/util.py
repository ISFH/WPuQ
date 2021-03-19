from datetime import date, datetime

import numpy as np
import pandas as pd


def month_year_iter(start_month, start_year, end_month, end_year):
    '''
    Creates an iterator over months and years

    Parameters
    ----------
    start_month : int
        Month to start from
    start_year : int
        Year to start from
    end_month : int
        The month to end at
    end_year : int
        The year to end at

    Yields
    ------
    y : int
        The year.
    m : int
        The month.
    '''
    ym_start = 12 * start_year + start_month - 1
    ym_end = 12 * end_year + end_month
    for ym in range(ym_start, ym_end):
        y, m = divmod(ym, 12)
        yield y, m+1


def nround(x, base=5):
    '''
    Rounds a number to a given base

    Parameters
    ----------
    x : float
        The number to round
    base : int, optional
        The base to to round to. The default is 5

    Returns
    -------
    float
        The rounded number.

    '''
    return [base * round(elem / base) for elem in x]


def rreplace(s, old, new, occurence):
    '''
    Replaces a substring in a string for a limited amount of times, starting
    from the back.

    Parameters
    ----------
    s : str
        The full string
    old : str
        The substring that should be replaced
    new : str
        The substring that replaces old
    occurence : int
        The maximum number of times to replace the substring

    Returns
    -------
    s_new : str
        The replaced string
    col_name : str
        Fill

    '''
    li = s.rsplit(old, occurence)
    s_new = li[0]
    col_name = new.join(li[1:])
    return s_new, col_name


def get_season(now):
    '''
    Gets the season from a date

    Parameters
    ----------
    now : datetime.datetime
        The date

    Returns
    -------
    str

    '''
    Y = 2000
    seasons = [('Winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('Spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('Summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('Autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('Winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def get_devices():
    '''
    Gets the parameters of the measurement devices.

    Returns
    -------
    devices : pd.DataFrame
        The device infos

    '''
    devices = pd.read_excel('measurement_devices.xlsx', header=0,
                                     skiprows=5, usecols="A:P")
    devices.dropna(how='all', axis=0, inplace=True)
    devices['Messstelle_Medium'].fillna(method='ffill', inplace=True)
    devices['Hersteller_Typ'].fillna(method='ffill', inplace=True)
    return devices


def get_parameters_from_device(devices, names, tags, feed, feed_names):
    '''
    Gets key parameters from a device

    Parameters
    ----------
    devices : pd.DataFrame
        A dataframe containing all devices and their parameters
    names : dict
        A processed dict of obj['username']
    tags : dict
        A processed dict of feed['tag']
    feed : dict
        Description of the feed
    feed_names : dict
        A processed dict of feed['name']

    Returns
    -------
    lb : float
        The minimum value which the measurement device can measure
    ub : float
        The maximum value which the measurement device can measure
    err_margin : str
        The error margin of the measurement device
    measure_range : str
        The measurement range of the measurement device
    measure_device : str
        The name of the measurement device
    unit : str
        The unit of the measurement

    '''
    measure_point = (names[0][0] + '_' + names[1][0] + '_' + tags[0][0]
             + '_' + tags[1][0] + '_' + tags[2][0])
    row = devices.loc[
        (devices['Messstelle_Medium'] == measure_point)
        & (devices['Feed'] == feed_names[0])
        & (devices['Unnamed: 7'] == feed_names[1])]
    # measurement devices are only known for Ohrberg, rest is empty
    if row.empty:
        lb = 0
        ub = 0
        err_margin = 0
        measure_range = 0
        measure_device = 0
        unit = ''
    else:
        lb = row['Min'].values[0]
        ub = row['Max'].fillna(np.inf).values[0]
        err_margin = row['Fehlergrenze'].values[0]
        measure_range = ','.join([str(lb), str(ub)])
        measure_device = row['Hersteller_Typ'].values[0]
        unit = row['Einheit'].values[0]
    return lb, ub, err_margin, measure_range, measure_device, unit


def fill_outliers(profile, n_hours=None, fill=True):
    '''
    Detects and fills outliers in the measurements. According to the matlab
    function filloutliers: An outlier is an element that is greater than 3
    scaled median absolute deviation (MAD) away from the median.

    Parameters
    ----------
    profile : pandas.DataFrame
        The data to correct
    n_hours : int in hours, optional
        If None, the median is calculated as a global median. If not None,
        the median is calculated in n_hours intervalls.
        The default is None.
    fill : bool, optional
        Returns the filled profile if fill is True. Returns the detected
        outliers if fill is False. The default is True

    Returns
    -------
    new_profile : pandas.DataFrame
        The corrected profile
    outliers : pandas.Index
        The index of the outliers
    '''
    new_profile = profile.copy()
    # there are occurences where the full profile is empty
    if new_profile['real'].isna().all():
        return new_profile['real'].fillna(0)
    if n_hours:
        # build the median per n_hours
        new_profile['date'] = new_profile.index.date
        new_profile['nhour'] = nround(new_profile.index.hour, n_hours)
        median = (new_profile.groupby(['date', 'nhour']).median()
                  .reset_index())
        new_profile = new_profile.merge(median, on=['date', 'nhour'],
                                        suffixes=('_real', '_median'))
        new_profile.columns = [col.split('_')[-1] if len(col.split('_'))
                               > 1 else col for col in new_profile.columns]
        # build the median of the difference to the median per n_hours
        new_profile['diff_to_median'] = abs(
            new_profile['real'] - new_profile['median'])
        median = new_profile[['date', 'nhour', 'diff_to_median']].groupby(
            ['date', 'nhour']).median().reset_index()
        new_profile = new_profile.merge(median, on=['date', 'nhour'],
                                        suffixes=('', '_median'))
        # identify rows where diff_to_median > 3 * diff_to_median_median
        outliers = new_profile[new_profile['diff_to_median'] >
                    3 * new_profile['diff_to_median_median']].index
    else:
        new_profile['median'] = new_profile.median().values[0]
        new_profile['diff_to_median'] = abs(
            new_profile['real'] - new_profile['median'])
        outliers = new_profile[new_profile['diff_to_median'] >
                    3 * new_profile['median']].index
    if fill:
        # correct these rows with pchip
        new_profile['corrected'] = new_profile['real'].copy()
        new_profile.loc[outliers, 'corrected'] = np.NaN
        new_profile['corrected'].interpolate(method='pchip', inplace=True)
        new_profile.index = profile.index
        return new_profile['corrected']
    else:
        return outliers


def correct_from_device_parameters(profile, name, feed_names, tags, lb, ub):
    '''
    Corrects profile values from the upper and lower bound of the measure
    device.

    Parameters
    ----------
    profile : pd.DataFrame or pd.Series
        The profile to correct
    name : dict
        Name of the feed. Assumes that profile is a Series if name is None
    feed_names : dict
        A processed dict of feed['name']
    tags : dict
        A processed dict of feed['tag']
    lb : float
        The lower bound
    ub : float
        The upper bound

    Returns
    -------
    profile : pd.DataFrame
        The adjusted profile

    '''
    
    if name:
        # correct values greater than the upper bound to nan
        profile.loc[profile[name] > ub, name] = np.NaN
        # correct values smaller than the lower bound to nan. Exceptions:
        # Lower bound is greater than 0 or the feed is heatpump power
        if (lb > 0) or ((tags[0][0] == 'HEATPUMP')
                        & (feed_names[0] == 'POWER')):
            profile.loc[profile[name] < lb, name] = 0
        else:
            profile.loc[profile[name] < lb, name] = np.NaN
        # now fill missing values with the nearest valid entry
        profile[name] = profile[name].interpolate(method='nearest')
    else:
        # correct values greater than the upper bound to nan
        profile.loc[profile > ub] = np.NaN
        # correct values smaller than the lower bound to nan. Exceptions:
        # Lower bound is greater than 0 or the feed is heatpump power
        if (lb > 0) or ((tags[0][0] == 'HEATPUMP')
                        & (feed_names[0] == 'POWER')):
            profile.loc[profile < lb] = 0
        else:
            profile.loc[profile < lb] = np.NaN
        # now fill missing values with the nearest valid entry
        profile = profile.interpolate(method='nearest')
    return profile


def harmonize_timestamp(profile, start_date, end_date, limit=8640):
    '''
    Measurement data might have missing values in the timestamps.
    Harmonizes the data to 10s timestamps.

    Parameters
    ----------
    profile : pandas.DataFrame
        The profile to adjust
    start_date : datetime.datetime
        The desired start date
    end_date : datetime.datetime
        The desired end date
    limit : int (10s)
        The maximum number of consecutive NaNs that will be filled by fillna
        and interpolation. The default is 8640 (equivalent to 1 day)

    Returns
    -------
    profile : pandas.DataFrame
        The adjusted profile

    '''
    # profiles should always contain full months in 10s steps. Create an
    # index and resample to that index to make sure this is fulfilled
    t_index = pd.DatetimeIndex(
        pd.date_range(start=start_date, end=end_date, freq='10s'))
    # linear interpolation and fill leading nans with 0s
    profile = profile.resample('10s').mean().reindex(t_index)
    profile = (profile.interpolate(method='linear', limit=limit)
               .fillna(method='ffill', limit=limit)
               .fillna(method='bfill', limit=limit))
    return profile
