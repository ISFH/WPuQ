from datetime import datetime as dt
from calendar import month_abbr
import datetime
import os

import h5py
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from .h5ls import H5ls
from .quarters import pv_objs
from .util import get_season
from .slpe import ElecSlp

font = {'size': 20}

matplotlib.rc('font', **font)

idx = pd.IndexSlice



def tz_localize(index):
    index = index.tz_localize(
        'Europe/London', nonexistent='shift_forward', ambiguous='NaT')
    index = index.tz_convert('Europe/Berlin')
    return index



class WPuQPlots():
    '''
    This class generates plots from the WPuQ data.
    
    Parameters
    ----------

    Attributes
    ----------
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

        '''
        self.pv_objs = pv_objs[quarter]

    def validate_power_vs_energy(self, folder, n_plot=5, current=None,
                                 plot_overview=True, non_negs=True,
                                 non_missing=True, non_inf=True,
                                 drop_dups=False):
        '''
        Generates plots of annual power vs energy from the xlsx validation
        files.

        Parameters
        ----------
        folder : str
            The folder where the excel validation and hdf5 raw files are
            stored. Is used to store the plots as well.
        n_plot : int, optional
            Defines the number of plots. The function takes the upper and lower
            amount of n_plot from the validation file to generate plots. The
            default is 5.
        current : str, optional
            Specifies if only a certain current type (active/reactive/apparent)
            should be considered. The default is None.
        plot_overview : bool, optional
            Specifies if an overview of the calculated differences is plotted.
            The default is True.
        non_negs : bool, optional
            The validation can have negative values which means that either
            power or energy sum over the year is negative. This is usually due
            to PV feedin and not necessarily an error. Setting non_negs to True
            additionally plots the lower amount of n_plot without considering
            negatives. The default is True.
        non_missing : bool, optioal
            Some objects are missing data from certain months due to failures
            or changes to the measurement devices. Setting non_missing to True
            drops failed objects from the evaluation, because deviations in
            annual results are then to be expected. The default is True.
        non_inf : bool, optional
            Some objects produce NaNs, meaning that the annual energy is 0.
            This is irrelevant in most of the times. Setting non_inf to True
            excludes these objects from the plots. The default is True.

        Returns
        -------

        '''
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
              + ' Validating power vs. energy.')
        folder_val = os.path.join(folder, 'validation')
        folder_res = os.path.join(folder, 'resampled')
        results = pd.read_excel(
            os.path.join(folder_val, 'ys_rel_per_node.xlsx'))
        results[0] /= 1e3
        if current:
            results = results[results['type'] == current]
        if non_missing:
            missing = pd.read_excel(
                os.path.join(folder_val, 'validation_months_available.xlsx'),
                index_col=0)
            # 2018 is missing january to april, because the measurements started
            # late
            if '2018' in folder:
                from ast import literal_eval
                missing = pd.DataFrame(missing['missing'].apply(
                    lambda x: str([elem for elem in literal_eval(x)
                               if elem not in [1, 2, 3, 4]])))
            missing = missing[missing['missing'] != '[]']
            results = results[~results['object'].isin(missing.index)]
        if non_inf:
            results = results[results[0] != np.float('inf')]
        # plot overview
        if plot_overview:
            for feed in ['TRANSFORMER', 'HOUSEHOLD', 'HEATPUMP']:
                for direction in ['IMPORT', 'EXPORT']:
                    self.plot_overview_pe_bar(
                        results, current=current, feed=feed, direction=direction,
                        filename='OVERVIEW_P_VS_E', strfile=folder_val)
        # plot first n rows
        if drop_dups:
            first = results.drop_duplicates(
                subset=['object', 'feed'], keep='first').iloc[:n_plot]
        else:
            first = results.iloc[:n_plot]
        for idx, row in first.iterrows():
            self.plot_power_vs_energy(
                obj=row['object'], feed=row['feed'], current=row['type'],
                phase=row['phase'], res='5min', folder=folder_res,
                strfile=folder_val)
        # plot last n rows
        if drop_dups:
            last = results.drop_duplicates(
                subset=['object', 'feed'], keep='last').iloc[n_plot * -1:]
        else:
            last = results.iloc[n_plot * -1:]
        for idx, row in last.iterrows():
            self.plot_power_vs_energy(
                obj=row['object'], feed=row['feed'], current=row['type'],
                phase=row['phase'], res='5min', folder=folder_res,
                strfile=folder_val)
        if non_negs:
            last = results[results[0] > 0].iloc[n_plot * -1:]
            for idx, row in last.iterrows():
                self.plot_power_vs_energy(
                    obj=row['object'], feed=row['feed'], current=row['type'],
                    phase=row['phase'], res='5min', folder=folder_res,
                    strfile=folder_val)

    def plot_data_quality(self, folder, quarter, years, feed, power,
                          objs=None, strfile=None):
        '''
        Plots the data quality saved in the memento.hdf5 file.

        Parameters
        ----------
        folder : str
            The top level data folder where subfolders containing data per
            quarter and year can be found.
        quarter : str
            The quarter to plot
        years : list of integers
            The years to plot
        feed : str
            The feed to plot. Options are 'HOUSEHOLD' and 'HEATPUMP'
        power : str
            Options are 'POWER' and 'ENERGY'. Decides if power or energetic
            measurements are plotted
        objs : list
            A list of objects to plot. Plots all objects if None.
            The default is None.
        strfile : str, optional
            The directory or filename of the plot. If it is a filename,
            filename is overwritten by strfile. If it is a directory, the plot
            is stored in this directory. If it is None, the plot is shown and
            not saved. The default is None.

        Returns
        -------

        '''
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
              + ' Plotting the data quality.')
        dsets = ['ELECTRICITY_POWER_TOTAL', 'ELECTRICITY_ENERGY_TOTAL_IMPORT']
        start_y = years[0]
        end_y = years[-1] + 1
        # 2018 measurements started in May
        if start_y == 2018:
            start = f'05-01-{start_y}'
        else:
            start = f'01-01-{start_y}'
        df = pd.DataFrame(index=pd.DatetimeIndex(pd.date_range(
            start=start, end=f'01-01-{end_y}', freq='10s')))
        for year in years:
            memento_filename = os.path.join(
                folder, quarter + '_' + str(year), 'resampled', 'memento.hdf5')
            memento_file = h5py.File(memento_filename, 'r')
            visitor = H5ls()
            memento_file.visititems(visitor)
            dset_names = visitor.names
            for dset_name in dset_names:
                if objs:
                    if not any(d in dset_name for d in objs):
                        continue
                if not any(d in dset_name for d in dsets):
                    continue
                if (feed not in dset_name) or (power not in dset_name):
                    continue
                dset_name = dset_name.replace('/table', '')
                print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + '\t' + 
                      str(year) + ' ' + dset_name)
                obj = dset_name.split('/')[0]
                reason = dset_name.split('/')[-1]
                profile = pd.read_hdf(memento_filename, dset_name)
                profile.index = pd.to_datetime(profile.index, unit='s')
                df.loc[df.index.intersection(profile.values), obj] = reason
            # some objects are missing data for the whole year, but 2018 is
            # complete fortunately. We can therefore detect objects missing
            # data first and then just fillna all remaining values to available
            year_df = df.loc[df.index.year == year].copy()
            na_cols = year_df.loc[:, year_df.isna().all()].columns.to_list()
            df.loc[df.index.year == year, na_cols] = 'missing'
        df = df.fillna('available')
        str_to_numeric = {
            'device_range': 0.,
            'outliers': 1,
            'missing': 2,
            'timestamps': 3,
            'available': 4}
        color_dict = {'device_range': 'darkolivegreen',
                  'outliers': 'gold',
                  'missing': 'maroon',
                  'timestamps': 'white',
                  'available': 'midnightblue'}
        entries = set(np.concatenate(
            [df[col].unique() for col in df.columns]).ravel())
        colors = [color_dict[k] for k, v in str_to_numeric.items()
                  if k in entries]
        df = df.replace(str_to_numeric)
        prcnt_avail = df.apply(pd.value_counts).fillna(0).loc[4] / len(df)
        df = df.resample('3min').min().T
        # sort the rows alphanumerically by building number
        df['building_nr'] = df.index.str.split('(\d+)').str[1].astype(int)
        df = df.sort_values(by='building_nr', ascending=False).drop(
            'building_nr', axis=1)
        prcnt_avail = prcnt_avail.reindex(df.index)
        df.columns = pd.to_datetime(df.columns)
        # plot
        fig, (ax2, ax) = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            gridspec_kw={'width_ratios': [1, 3]},
            figsize=(12, max(3, len(df) / 2)))
        # plot right side (showing the matrix)
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = [v for k, v in str_to_numeric.items() if k in entries]
        bounds.append(max(bounds) + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        c = ax.pcolor(df, cmap=cmap, norm=norm)
        # adjust right side xticks to show month-year
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        ax2.set_ylim(bottom=0, top=len(df.index))
        xticks = [f'{month_abbr[x.month]}-{x.year}' for x
                  in df.T.resample('M').mean().index.to_period('M')]
        xticks = xticks[0::2]
        ax.xaxis.set_major_locator(plt.LinearLocator(len(xticks)))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(xticks))
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', rotation=90)
        # plot left side (showing percent available)
        ypos = ax.get_yticks()
        ax2.barh(ypos, prcnt_avail)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(df.index)
        # annotate values to left side
        for axis, value in zip(ax2.patches, prcnt_avail):
            ax2.text(axis.get_width() / 3, axis.get_y() + 0.25,
                     str(round(value * 100, 1)) + '%')
        ax2.tick_params(labelbottom=False)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Data availability')
        ax2.xaxis.set_label_position('top')
        # adjust colorbar to function as a legend
        cb = plt.colorbar(c)
        labels = [k.title() for k, v in str_to_numeric.items() if k in entries]
        loc = [v + 0.5 for k, v in str_to_numeric.items() if k in entries]
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        plt.subplots_adjust(wspace=0.05)
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(
                    strfile, feed + '_' + power + '_months_available.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_overview_pe_bar(self, results, feed=None, phase=None, current=None,
                      direction=None, filename='', strfile=None):
        '''
        Plots an overview of the annual mean of power vs energy measurements
        as a bar plot.

        Parameters
        ----------
        results : pd.DataFrame
            The dataframe containing the results to plot. It has to contain the
            ratios of power / energy per object, feed, type, phase and
            direction.
        feed : str, optional
            The feeds to plot. Plots all feeds if None. The default is None.
        phase : str, optional
            The phase to plot. Plots all phases if None. The default is None.
        current : str, optional
            The current to plot. Plots all currents if None. The default is
            None.
        direction : str
            The direction to plot. The default is None.
        filename : str, optional
            The filename for the plot. Gets extended by current, feed and
            phase. The default is ''.
        strfile : str, optional
            The directory or filename of the plot. If it is a filename,
            filename is overwritten by strfile. If it is a directory, the plot
            is stored in this directory. If it is None, the plot is shown and
            not saved. The default is None.
        Returns
        -------

        '''
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Plotting overview.')
        color_dict = {
            'P': 'green',
            'Q': 'red',
            'S': 'blue'}
        columns = list(results.columns.copy())
        suptitle = 'Overview'
        if current:
            suptitle += f' for current {current}'
            filename += f'_{current}'
        if feed:
            results = results[results['feed'] == feed]
            columns.remove('feed')
            suptitle += f' for feed {feed}'
            filename += f'_{feed}'
        if phase:
            results = results[results['phase'] == phase]
            columns.remove('phase')
            suptitle += f' for phase {phase}'
            filename += f'_{phase}'
        if direction:
            results = results[results['direction'] == direction]
            columns.remove('direction')
            suptitle += f' for direction {direction}'
            filename += f'_{direction}'
        if results.empty:
            return
        colors = [color_dict[col] for col in results['type']]
        columns.remove('type')
        results = results[columns]
        columns.remove(0)
        results.index = results[columns].apply(lambda x: '_'.join(x), axis=1)
        
        fig, ax = plt.subplots(figsize=(21, 9))
        results[0].plot(ax=ax, kind='bar', color=colors)
        if 'TS' in filename:
            ax.set_ylabel('P / (U * I * cosphi)')
        else:
            ax.set_ylabel('Power / Energy')
        ylim_df = results.replace([np.inf, -np.inf], np.nan).dropna()
        if ylim_df[0].max() - ylim_df[0].min() < 2:
            ax.set_ylim(ylim_df[0].min() - 0.1, ylim_df[0].max() + 0.01)
        else:
            ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True)
        ax.tick_params(axis='x', which='major', labelsize=12)
        fig.suptitle(suptitle)
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, filename + '.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_power_vs_energy(self, folder, obj, feed, current, phase,
                             res='10s', strfile=None):
        '''
        Plots cumulated power vs. energy for a given feed.

        Parameters
        ----------
        folder : str
            The folder where the hdf5 file containing the feed is stored.
        obj : str
            The object to plot
        feed : str
            The feed to plot
        current : str
            The current to plot
        phase : str
            The phase to plot
        res : str, optional
            The temporal resolution to plot. The default is '10s'.
        strfile : str, optional
            The path where the plot should be stored. If strfile is a directory,
            the plot is stored in that directory under an auto-generated name.
            If strfile is None, the plot is shown instead of saved. The default
            is None.

        Returns
        -------

        '''
        filename = '_'.join([obj, feed, current, phase])
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + f' Plotting {filename}.')
        translation = dict(
            power=dict(
                S='S',
                P='P',
                Q='Q'),
            energy=dict(
                S='SE',
                P='E',
                Q='QE')
            )
        source_file_energy = os.path.join(folder, 'energy_data_10s.hdf5')
        source_file_power = os.path.join(folder, 'data_10s.hdf5')
        if obj == 'ES1':
            direction = 'OUT'
            subset = 'MISC'
        else:
            direction = 'IN'
            if int(obj.replace('SFH', '')) in self.pv_objs:
                subset = 'WITH_PV'
            else:
                subset = 'NO_PV'
        dset_name = f'{subset}/{obj}/{feed}'
        # read energy
        energy = pd.DataFrame(columns=['IMPORT', 'EXPORT'])
        trans = translation['energy'][current]
        for direct in ['IMPORT', 'EXPORT']:
            col_name = f'{trans}_{phase}_{direct}'
            try:
                profile = pd.read_hdf(source_file_energy, dset_name)
                profile.index = pd.to_datetime(profile.index, unit='s')
                energy[direct] = profile[col_name]
            except KeyError:
                energy[direct] = 0
        energy = energy.fillna(0)
        energy['IMPORT'] = energy['IMPORT'] - energy.iloc[0, 0]
        energy['EXPORT'] = (energy['EXPORT'] - energy.iloc[0, 1]) * -1
        # read power
        trans = translation['power'][current]
        power = pd.DataFrame(columns=['IMPORT', 'EXPORT'], index=energy.index)
        col_name = f'{trans}_{phase}'
        tmp = pd.read_hdf(source_file_power, dset_name)
        tmp.index = pd.to_datetime(tmp.index, unit='s')
        if current == 'S':
            dset_name = f'{feed}/{direction}/ELECTRICITY_POWER_FACTOR_{phase}'
            pf = tmp[f'PF_{phase}']
            pf[pf < 0] = -1
            pf[pf > 0] = 1
            tmp = tmp[col_name] * pf
        else:
            # with pv objects have different column names. Fix it
            if col_name == 'P_TOT' and 'P_TOT_WITH_PV' in tmp.columns:
                tmp = tmp['P_TOT_WITH_PV']
            else:
                tmp = tmp[col_name]
        power['EXPORT'] = (tmp[tmp < 0].cumsum() / 1e3 / 360).reindex(
            power.index).ffill().fillna(0)
        power['IMPORT'] = (tmp[tmp > 0].cumsum() / 1e3 / 360).reindex(
            power.index).ffill().fillna(0)
        # plot
        fig, ax = plt.subplots(figsize=(16, 9))
        power['IMPORT'].resample(res).mean().plot(
            ax=ax, color='blue', label='Power Import')
        power['EXPORT'].resample(res).mean().plot(
            ax=ax, color='blue', label='Power Export', linestyle='--')
        energy['IMPORT'].resample(res).mean().plot(
            ax=ax, color='orange', label='Energy Import')
        energy['EXPORT'].resample(res).mean().plot(
            ax=ax, color='orange', label='Energy Export', linestyle='--')
        ax.set_ylabel('Cumulated Energy in kWh')
        plt.legend()
        fig.suptitle('_'.join([obj, feed, current, phase]))
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, filename + '.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_timeseries(self, folder, obj, feed, phase, res='10s',
                        snapshot=None, annotate=False, strfile=None):
        '''
        Plots meausurements of timeseries.

        Parameters
        ----------
        folder : str
            The folder where the hdf5 file containing the feed is stored.
        obj : str
            The object to plot
        feed : str
            The feed to plot
        phase : str
            The phase to plot
        res : str, optional
            The temporal resolution to plot. The resolution has to be available
            in the resampled folder. The default is '10s'.
        snapshot : str
            Plots the whole timeseries if None. If 'min', plots a snapshot
            around the point with the smallest ratio between indiv and
            combined. If 'max', plots a snapshot around the hightest ratio.
            The default is None.
        annotate : bool
            Annotates a text info of how many values seem to show errors if
            True. The default is False.
        strfile : str, optional
            The path where the plot should be stored. If strfile is a directory,
            the plot is stored in that directory under an auto-generated name.
            If strfile is None, the plot is shown instead of saved. The default
            is None.

        Returns
        -------

        '''
        if snapshot:
            filename = '_'.join([obj, feed, phase, snapshot.split('_')[0]])
        else:
            filename = '_'.join([obj, feed, phase])
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + f' Plotting {filename}.')
        translation = dict(
            combined=['P'],
            individual=['U', 'I', 'PF'],
            units='W*'
            )
        source_file = os.path.join(folder, 'data_' + res + '.hdf5')
        # read timeseries
        dset_name = f'{obj}/{feed}'
        if any(obj in dset_name for obj in [
                            'ES1', 'PV1', 'WEATHER_ISFH']):
            dset_name = 'MISC/' + dset_name
        elif 'SFH' in dset_name:
            obj_nr = int(dset_name.split('/')[0][3:])
            if obj_nr in self.pv_objs:
                dset_name = 'WITH_PV/' + dset_name
            else:
                dset_name = 'NO_PV/' + dset_name
        # power measurement
        cols_c = [c + '_' + phase for c in translation['combined']]
        combined = pd.read_hdf(source_file, dset_name, columns=cols_c)
        combined.index = pd.to_datetime(combined.index, unit='s')
        combined = combined.sum(axis=1)
        # individual measurements of I and U
        if phase == 'TOT':
            cols_i = [c + '_' + str(p) for c in translation['individual']
                      for p in [1, 2, 3]]
        else:
            cols_i = [c + '_' + phase for c in translation['individual']]
        indiv = pd.read_hdf(source_file, dset_name, columns=cols_i)
        indiv.index = pd.to_datetime(indiv.index, unit='s')
        indiv.columns = indiv.columns.str.split('_', expand=True)
        indiv = indiv.xs('U', axis=1, level=0).mul(
            indiv.xs('I', axis=1, level=0), axis=1).mul(
            indiv.xs('PF', axis=1, level=0), axis=1).sum(axis=1)
        ratio = (combined / indiv)
        # plot
        perc = round(
            len(ratio[(ratio > 3) | (ratio < 1 / 3)]) / len(ratio) *  100, 4)
        fig, ax = plt.subplots(figsize=(16, 9))
        ax2 = ax.twinx()
        if not snapshot:
            combined.abs().plot(ax=ax, color='orange',
                          label=translation['combined'][0])
            indiv.plot(ax=ax, color='darkviolet',
                       label=' * '.join(translation['individual']))
            ratio.plot(ax=ax2, color='black', label='Ratio',
                                    linewidth=0.5, linestyle='dashed')
        else:
            if 'max' in snapshot:
                timestamp = ratio.nlargest(int(snapshot.split('_')[1])).index
                timestamp = timestamp[-1]
            elif 'min' in snapshot:
                timestamp = ratio.nsmallest(int(snapshot.split('_')[1])).index
                timestamp = timestamp[-1]
            timestamp = ratio.index.get_loc(timestamp)
            filename = filename + '_' + str(timestamp)
            ts_low = max(timestamp - 25, 0)
            ts_high = min(timestamp + 25, len(combined))
            combined.iloc[ts_low:ts_high].abs().plot(
                ax=ax, color='orange',
                label=translation['combined'][0])
            indiv.iloc[ts_low:ts_high].plot(
                ax=ax, color='darkviolet',
                label=' * '.join(translation['individual']))
            ratio.iloc[ts_low:ts_high].plot(
                ax=ax2, color='black', label='Ratio', linewidth=0.5,
                linestyle='dashed')
        if annotate:
            ax.annotate(s=f'{str(perc)}% of values have\na ratio > 3 or < 1/3',
                        xy=(0.1, 0.6), xycoords='figure fraction', fontsize=16)
        ax.set_ylabel('Measured Power in W')
        ax2.set_ylabel('Ratio in -')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        #fig.suptitle('_'.join([obj, feed, phase]))
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, filename + '.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


    def plot_seasonal_load_curves(
            self, folder, objects, feed, res, sum_dsets=False,
            correct_pv=True, plot_slp=False, use_average=False,
            figtitle=False):
        '''
        Plot seasonal load curves

        Parameters
        ----------
        folder : str
            The folder containing the resampled datafiles
        objects : list
            A list containing the objects to plot. Plots the spatially
            aggregated load curve if None
        feed : str
            The feed to plot. Has to be one of ('HOUSEHOLD', 'HEATPUMP')
        res : str
            The temporal resolution to plot
        sum_dsets : bool
            Plots the different datasets in seperate plots if this is False.
            Builds a sum over all datasets and plots it in a single plot if
            this is True. The default is False.
        correct_pv : bool
            If the dataset contains houses with PV production and this is
            False, this function plots both the measured load, which is real
            load minus the PV production and can therefore be negative during
            PV feed-in, and an estimated load curve, calculated by the measured
            load plus an estimated PV feed-in. The default is True.
        plot_slp : bool
            Decides if a standard load profile is plotted with the measured
            load curve for validation reasons. This is the SLP H0 of BDEW for
            the household and the SLP of the Stadtwerke München for the heat
            pump. The default is False.
        use_average : bool
            Decides if the summed load curve or the average per household is
            plotted. This is only used if objects is None. The default is False.
        figtitle : bool
            Decides if the figure has a title. The default is False.
            

        Returns
        -------

        '''
        season_dict = {
            'Spring': 0,
            'Summer': 1,
            'Autumn': 2,
            'Winter': 3
            }
        day_dict = {
            'Weekday': 0,
            'Saturday': 1,
            'Sunday': 2
            }
        if not objects:
            filename = os.path.join(folder, 'data_spatial.hdf5')
            file = h5py.File(filename, 'r')
            dset_names = ['NO_PV/' + res + '/' + feed,
                          'WITH_PV/' + res + '/' + feed]
        elif objects == 'all':
            filename = os.path.join(folder, f'data_{res}.hdf5')
            file = h5py.File(filename, 'r')
            visitor = H5ls()
            file.visititems(visitor)
            dset_names = [name.replace('/table', '') for name in visitor.names
                          if feed in name]
        else:
            filename = os.path.join(folder, f'data_{res}.hdf5')
            file = h5py.File(filename, 'r')
            dset_names = []
            for obj in objects:
                dset_names.extend(['NO_PV/' + obj + '/' + feed])
                dset_names.extend(['WITH_PV/' + obj + '/' + feed])
                dset_names.extend(['MISC/' + obj + '/' + feed])
        for dset_name in dset_names:
            try:
                profile = pd.read_hdf(filename, dset_name)
                profile.index = pd.to_datetime(profile.index, unit='s')
            except KeyError:
                continue
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Plotting ' +
                  f'seasonal load curves for dset {dset_name}.')
            if correct_pv or 'P_TOT_WITH_PV' not in profile.columns:
                profile = profile['P_TOT']
            else:
                profile = profile[['P_TOT', 'P_TOT_WITH_PV']]
            if sum_dsets:
                if dset_name == dset_names[0]:
                    profile_tot = pd.DataFrame(profile.copy()).sum(axis=1)
                else:
                    profile_tot = profile_tot + pd.DataFrame(profile).sum(
                        axis=1)
                if dset_name != dset_names[-1]:
                    continue
                else:
                    dset_name = 'WITH_AND_NO_PV' + res + '/' + feed
                    profile = profile_tot
            if use_average:
                try:
                    n_objs = len(
                        file[dset_name + '/table'].attrs['objects_included'])
                    profile = profile / n_objs
                except KeyError:
                    pass
                if 'WITH_AND_NO_PV' in dset_name:
                    n_objs = 0
                    for dn in dset_names:
                        n_objs += len(
                            file[dn + '/table'].attrs['objects_included'])
                    profile = profile / n_objs
            profile.index = tz_localize(profile.index)
            day_name = profile.index.day_name()
            day_name = [dn if dn in ['Saturday', 'Sunday'] else 'Weekday'
                        for dn in day_name]
            # each day has the same season
            season = [get_season(index) for index in profile.index[::8640]]
            season = np.repeat(season, 8640)
            year = int(profile.index.year[0])
            if plot_slp and not 'WITH_PV' in dset_name:
                e_slp = ElecSlp(year)
                if feed == 'HOUSEHOLD':
                    slp_name = 'h0'
                elif feed == 'HEATPUMP':
                    slp_name = 'BDEW HP'
                ann_cons = profile.sum() / 360
                slp = e_slp.get_profile({slp_name: ann_cons * 4})[slp_name]
                print(f'Annual consumption: {ann_cons / 1e3} kWh')
                slp = slp.groupby(
                    [day_name[::90], season[::90], slp.index.time]).mean()
                slp = slp.unstack(2)
                slp.columns = pd.date_range(
                    f'{year}-01-01 00:00:00', f'{year}-01-01 23:59:50',
                    freq='15min')
            profile = profile.groupby(
                [day_name, season, profile.index.time]).mean()
            # setting the xticks later does not work unless the columns are a
            # real datetime object
            profile.index = profile.index.set_levels(
                pd.date_range(
                    f'{year}-01-01 00:00:00', f'{year}-01-01 23:59:50',
                    freq='10s'), level=2)
            profile = profile.unstack(2)

            fig, axes = plt.subplots(figsize=(12, 9), ncols=4, nrows=3,
                                     sharex=True, sharey=True)
            # mi columns means we have two profiles (with and without PV)
            if isinstance(profile.columns, pd.MultiIndex):
                for col in ['P_TOT', 'P_TOT_WITH_PV']:
                    if col == 'P_TOT':
                        color = 'blue'
                    else:
                        color = 'orange'
                    for seasonday in profile.index:
                        ax = axes[day_dict[seasonday[0]],
                              season_dict[seasonday[1]]]
                        profile.loc[seasonday, col].plot(
                            ax=ax, legend=False, color=color, x_compat=True)
                red_patch = mpatches.Patch(
                    color='orange', label='Including PV production')
                blue_patch = mpatches.Patch(
                    color='blue', label='Excluding PV production')
                plt.legend(handles=[red_patch, blue_patch], loc='upper center',
                              bbox_to_anchor=(-1.5, -0.8), ncol=2)
            elif isinstance(profile.index, pd.MultiIndex):
                for seasonday in profile.index:
                    ax = axes[day_dict[seasonday[0]],
                              season_dict[seasonday[1]]]
                    profile.loc[seasonday, :].plot(
                        ax=ax, color='blue', x_compat=True)
                    if plot_slp and not 'WITH_PV' in dset_name:
                        slp.loc[seasonday, :].plot(
                            ax=ax, color='red', x_compat=True)
            # set y labels
            for weekday in profile.index.get_level_values(0).unique():
                # left side ticks
                axl = axes[day_dict[weekday], 0]
                axl.tick_params(axis='y', rotation=45)
                axl.yaxis.set_major_locator(mticker.MaxNLocator(3))
                axl.yaxis.set_minor_locator(plt.NullLocator())
                # right side label
                axr = axes[day_dict[weekday], 3]
                axrt = axr.twinx()
                axrt.set_ylabel(weekday, rotation=90)
                axrt.yaxis.set_major_formatter(plt.NullFormatter())
                axrt.yaxis.set_minor_locator(plt.NullLocator())
            # set column titles
            for season in profile.index.get_level_values(1).unique():
                ax = axes[0, season_dict[season]]
                ax.set_title(season)
            # set xticks and disable xlabel
            for ax in axes[2]:
                ax.set_xlabel('')
                cols = profile.columns.get_level_values(-1)
                ax.set_xlim(cols[0], cols[-1])
                # major ticks and labels
                hours = mdates.HourLocator(byhour=[1, 6, 11, 16, 21])
                h_fmt = mdates.DateFormatter('%H:%M')
                ax.xaxis.set_major_locator(hours)
                ax.xaxis.set_major_formatter(h_fmt)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,
                         ha='center', rotation_mode='default')
                # minor ticks
                hours_minor = mdates.HourLocator(interval=1)
                ax.xaxis.set_minor_locator(hours_minor)
            # common y axis labels
            fig.text(0.02, 0.5, 'Load in W', va='center',
                     rotation='vertical')
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            plt.subplots_adjust(hspace=0.017, wspace=0.01)
            if figtitle:
                if not objects:
                    fig.suptitle('All objects: ' + feed + ', ' +
                                 dset_name.split('/')[0])
                else:
                    fig.suptitle(dset_name.replace('/', '_'))
            if plot_slp and not 'WITH_PV' in dset_name:
                strfile = os.path.join(folder, dset_name.replace('/', '_')
                                       + '_slp_seasonal.png')
                red_patch = mpatches.Patch(
                    color='red', label=f'SLP {slp_name.upper()}')
                blue_patch = mpatches.Patch(color='blue', label='WPuQ')
                axes[0, 1].legend(handles=[red_patch, blue_patch],
                                  loc='upper left', ncol=1, fontsize=18)
            else:
                strfile = os.path.join(folder, dset_name.replace('/', '_')
                                       + '_seasonal.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()
        file.close()

    def plot_annual_consumption(self, folder, strfile=None):
        '''
        Plots the annual electricity consumption per object, seperated by
        household and heat pump.

        Parameters
        ----------
        folder : str
            The folder where the xlsx file containing the annual consumption
            is stored.
        strfile : str, optional
            The path where the plot should be stored. If strfile is a directory,
            the plot is stored in that directory under an auto-generated name.
            If strfile is None, the plot is shown instead of saved. The default
            is None.

        Returns
        -------

        '''
        feeds = ['HOUSEHOLD', 'HEATPUMP']
        df = pd.read_excel(os.path.join(
            folder, 'validation', 'ys_abs_per_node.xlsx'))
        df = df[df['feed'].isin(feeds)]
        df['building_nr'] = df['obj'].str.split('(\d+)').str[1].astype(int)
        df = df.sort_values(by='building_nr').drop('building_nr', axis=1)
        # plot
        fig, ax = plt.subplots(figsize=(16, 9))
        width = 0.8
        vals = [df.loc[df['feed'] == feed, 0].to_numpy() for feed in feeds]
        labels = df.loc[df['feed'] == 'HOUSEHOLD', 'obj'].to_numpy()
        n = len(vals)
        _labels = np.arange(len(labels))
        for i in range(n):
            ax.bar(_labels - width / 2. + i / float(n) * width, vals[i],
                    width=width / float(n), align='edge', label=feeds[i]
            )   
        plt.xticks(_labels, labels, rotation=90)
        ax.tick_params(axis='y', rotation=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_xlim(0 - width, len(vals[0]) - 1 + width)
        ax.set_ylabel('Annual consumption in kWh/a')
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
        plt.legend()
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, 'annual_consumption.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_daily_for_report(
            self, folder, language='de', ymax=None, strfile=None):
        '''
        Plots the active and apparent power of the transformer together with
        the outside temperature and the temperature in the district heating
        network for the WPuQ final report.

        Parameters
        ----------
        folder : TYPE
            The folder where the hdf5 files containing the data can be found
        language : str
            The language of the legend and axis titles.
            Must be one of {'de', 'en'}. The default is de
        strfile : str, optional
            The path where the plot should be stored. If strfile is a directory,
            the plot is stored in that directory under an auto-generated name.
            If strfile is None, the plot is shown instead of saved. The default
            is None.

        Returns
        -------

        '''
        translation = dict(
            vorlauf=dict(
                en='Network flow temperature',
                de='Netzvorlauftemperatur'
            ),
            außen=dict(
                en='Ambient temperature',
                de='Außentemperatur'
            ),
            blind=dict(
                en='Reactive energy',
                de='Blindleistung'
            ),
            wirk=dict(
                en='Active energy',
                de='Wirkleistung'
            ),
            y1=dict(
                en='Energy in kWh/day',
                de='Tagesgemittelte Leistung in kW'
            ),
            y2=dict(
                en='Temperature in °C',
                de='Temperatur in °C'
            )
        )
        # get transformer data
        source_file = 'data_60min.hdf5'
        dset_name = 'MISC/ES1/TRANSFORMER'
        columns = ['P_TOT', 'Q_TOT']
        transformer = pd.read_hdf(
            os.path.join(folder, source_file), dset_name, columns=columns)
        transformer.index = pd.to_datetime(transformer.index, unit='s')
        transformer.index = tz_localize(transformer.index)
        transformer = transformer.resample('D').mean() / 1e3
        if language == 'en':
            transformer = transformer * 24
        transformer[transformer < 0] = 0
        # get district heating data
        try:
            source_file = 'dh_grid.hdf5'
            dset_name = 'DH_GRID/IN/HEAT_TEMPERATURE_FLOW'
            dh = pd.read_hdf(os.path.join(folder, source_file), dset_name)
        except FileNotFoundError:
            source_file = 'DISTRICT_HEATING_GRID.hdf5'
            dset_name = 'DISTRICT_HEATING_GRID/IN/HEAT_TEMPERATURE_FLOW'
            parent = os.path.abspath(os.path.join(folder, os.pardir))
            dh = pd.read_hdf(os.path.join(parent, source_file), dset_name)
        dh.index = pd.to_datetime(dh.index, unit='s')
        dh.index = tz_localize(dh.index)
        dh = dh.resample('D').mean()
        # get temperature data
        dset_name = 'WEATHER_SERVICE/IN/WEATHER_TEMPERATURE_TOTAL'
        try:
            source_file = 'weather.hdf5'
            weather = pd.read_hdf(os.path.join(folder, source_file), dset_name)
        except FileNotFoundError:
            source_file = 'WEATHER_STATION_1.hdf5'
            parent = os.path.abspath(os.path.join(folder, os.pardir))
            weather = pd.read_hdf(os.path.join(parent, source_file), dset_name)
        weather.index = pd.to_datetime(weather.index, unit='s')
        weather.index = tz_localize(weather.index)
        weather = weather.resample('D').mean()
        # total data
        data = pd.concat([transformer, dh, weather], axis=1)
        data = data[data['P_TOT'].notna()]
        # plot
        fig, ax = plt.subplots(figsize=(16, 9))
        ax2 = ax.twinx()
        data[['Q_TOT', 'P_TOT']].plot(
            ax=ax, color=['navy', 'royalblue'], kind='area', stacked=True)
        data['TEMPERATURE:FLOW'].plot(
            ax=ax2, label=translation['vorlauf'][language], color='red')
        data['TEMPERATURE:TOTAL'].plot(
            ax=ax2, label=translation['außen'][language], color='yellow')
        lines, labels = ax.get_legend_handles_labels()
        labels = [translation['blind'][language], translation['wirk'][language]]
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        ax.set_ylabel(translation['y1'][language])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
        ax.set_xlabel('')
        ax2.set_ylabel(translation['y2'][language])
        # move ylim so that legend is visible
        ax2.set_ylim((-6, 35))
        if ymax:
            ax.set_ylim(top=ymax)
        year = data.index.year[100]
        ax.set_xlim([datetime.date(year, 1, 1), datetime.date(year, 12, 1)])
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(strfile, 'daily_plot_wpuq_report.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_annual_consumption_heat_pump_for_report(
            self, folder, method, language='de', strfile=None):
        '''
        Detects three operation modes (heat pump, heating rod and pumps only)
        from the heat pump load curve.

        Parameters
        ----------
        folder : str
            The home folder containing both the validation and resampled
            folder.
        method : str
            The method used to seperate heating rod from heat pump operation
        strfile : str
            Plots the data if a path to save a plot is given.
            The default is None.

        Returns
        -------

        '''

        translation = dict(
            comp=dict(
                en='Compressor operation',
                de='Kompressorbetrieb'),
            rod=dict(
                en='Heating rod operation',
                de='Heizstabbetrieb'),
            pumps=dict(
                en='Stand-by operation',
                de='Stand-By-Betrieb')
        )
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Detecting heat pump '
              'operation modes.')
        index = ['obj'] + [val[language] for key, val in translation.items()]
        total = pd.DataFrame(index=index)
        missing = pd.Series(name='missing')
        filename = os.path.join(folder, 'data_10s.hdf5')
        file = h5py.File(filename, 'r')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            if not 'HEATPUMP' in dset_name:
                continue
            obj = dset_name.split('/')[1]
            missing.loc[obj] = 1
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + '\t ' + obj)
            profile = pd.read_hdf(filename, dset_name).set_index('index')
            profile.index = pd.to_datetime(profile.index, unit='s')
            # mark buildings with nans. Special treatment for 2018
            if '2018' in folder:
                if profile.loc[profile.index.month > 5, 'P_TOT'].isna().any():
                    missing.loc[obj] = 0.5
            else:
                if profile['P_TOT'].isna().any():
                    missing.loc[obj] = 0.5
            profile = profile[['P_TOT', 'S_TOT', 'Q_TOT']]
            # assumption that a heat pump consumption larger than 4 kW
            # means heating rod operation
            if method == 'larger 4kW':
                profile.loc[profile['P_TOT'] < 4000, 'operation_mode'] = \
                    translation['comp'][language]
            # assumption that the heating rod is an ohmic resistor, meaning
            # that apparent and active power are equal if the heat pump
            # runs in heating rod operation mode
            elif method == 'Power Factor':
                profile.loc[
                    (profile['P_TOT'] > 100) & (profile['Q_TOT'] > 100),
                    'operation_mode'] = translation['comp'][language]
            # assumption that consumption < 100 W is pumps only
            profile.loc[profile['P_TOT'] < 100, 'operation_mode'] = \
                translation['pumps'][language]
            profile['operation_mode'].fillna(
                translation['rod'][language], inplace=True)
            profile = profile.groupby('operation_mode').sum()
            profile.loc['obj'] = obj
            total = pd.concat([total, profile['P_TOT']], axis=1)
        total = total.T.reset_index(drop=True)
        total = total.set_index('obj')
        total = total.merge(missing, left_index=True, right_index=True)
        total['building_nr'] = total.index.str.split('(\d+)').str[1].astype(int)
        total = total.sort_values(by='building_nr', ascending=True).set_index(
            'building_nr')
        cols = [val[language] for key, val in translation.items()]
        total[cols] = total[cols] / 360 / 1e6
        # plot
        colors = ['yellowgreen', 'orangered', 'gainsboro']
        fig, ax = plt.subplots(figsize=(16, 9))
        total.loc[total['missing'] == 1, cols].plot(
            ax=ax, kind='bar', stacked=True, color=colors)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='best')
        if language == 'de':
            ax.set_ylabel('Wirkenergie [MWh/a]')
            ax.set_xlabel('Gebäudenummer')
        elif language == 'en':
            ax.set_ylabel('Active energy [MWh/a]')
            ax.set_xlabel('Building number')
        ax.tick_params(axis='x', rotation=90)
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(
                    strfile, 'heat_pump_operation_wpuq_report.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            total.to_csv(strfile.replace('.png', '.csv'))
            plt.close()

    def plot_annual_consumption_household_for_report(
            self, folder, strfile=None):
        '''
        Plots the annual household consumption.

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

        '''

        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S")
              + ' Plotting annual household consumption.')
        total = pd.Series(name='Haushalt')
        missing = pd.Series(name='missing')
        filename = os.path.join(folder, 'resampled', 'data_10s.hdf5')
        file = h5py.File(filename, 'r')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            if not 'HOUSEHOLD' in dset_name:
                continue
            obj = dset_name.split('/')[1]
            missing.loc[obj] = 1
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + '\t ' + obj)
            profile = pd.read_hdf(filename, dset_name).set_index('index')
            profile.index = pd.to_datetime(profile.index)
            # mark buildings with nans. Special treatment for 2018
            if '2018' in folder:
                if profile.loc[profile.index.month > 5, 'S_TOT'].isna().any():
                    missing.loc[obj] = 0.5
            else:
                if profile['S_TOT'].isna().any():
                    missing.loc[obj] = 0.5
            total.loc[obj] = profile['S_TOT'].sum()
        total = pd.concat([total, missing], axis=1)
        total['building_nr'] = total.index.str.split('(\d+)').str[1].astype(int)
        total = total.sort_values(by='building_nr', ascending=True).set_index(
            'building_nr')
        cols = ['Haushalt']
        total[cols] = total[cols] / 360 / 1e6
        # plot
        colors = ['darkgrey']
        fig, ax = plt.subplots(figsize=(16, 9))
        total.loc[total['missing'] == 1, cols].plot(
            ax=ax, kind='bar', stacked=True, color=colors)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='best')
        ax.set_ylabel('Energie [MWh/a]')
        ax.set_xlabel('Gebäudenummer')
        ax.tick_params(axis='x', rotation=90)
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(
                    strfile, 'household_operation_wpuq_report.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            plt.close()

    def plot_annual_consumption_heat_pump_and_household_for_report(
            self, folder, language='de', strfile=None):
        '''
        Detects three operation modes (heat pump, heating rod and pumps only)
        from the heat pump load curve.

        Parameters
        ----------
        folder : str
            The home folder containing both the validation and resampled
            folder.
        method : str
            The method used to seperate heating rod from heat pump operation
        strfile : str
            Plots the data if a path to save a plot is given.
            The default is None.

        Returns
        -------

        '''

        translation = dict(
            pumps=dict(
                en='Stand-by operation',
                de='Stand-By-Betrieb'),
            comp=dict(
                en='Compressor operation',
                de='Kompressorbetrieb'),
            rod=dict(
                en='Heating rod operation',
                de='Heizstabbetrieb'),
            hh=dict(
                en='Household',
                de='Haushalt')
        )
        print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Detecting heat pump '
              'operation modes.')
        index = ['obj'] + [val[language] for key, val in translation.items()]
        total = pd.DataFrame(index=index)
        missing = dict()
        filename = os.path.join(folder, 'data_10s.hdf5')
        file = h5py.File(filename, 'r')
        visitor = H5ls()
        file.visititems(visitor)
        dset_names = visitor.names
        for dset_name in dset_names:
            if not 'HEATPUMP' in dset_name and not 'HOUSEHOLD' in dset_name:
                continue
            obj = dset_name.split('/')[1]
            feed = dset_name.split('/')[2]
            if not obj in missing:
                missing[obj] = dict()
            missing[obj][feed] = 1
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + '\t ' + obj)
            profile = pd.read_hdf(filename, dset_name).set_index('index')
            profile.index = pd.to_datetime(profile.index, unit='s')
            # mark buildings with nans. Special treatment for 2018
            if '2018' in folder:
                if profile.loc[profile.index.month > 5, 'P_TOT'].isna().any():
                    missing[obj][feed] = 0.5
            else:
                if profile['P_TOT'].isna().any():
                    missing[obj][feed] = 0.5
            profile = profile[['P_TOT', 'S_TOT', 'Q_TOT']]
            # assumption that a heat pump consumption larger than 4 kW
            # means heating rod operation
            if 'HEATPUMP' in dset_name:
                profile.loc[profile['P_TOT'] < 4000, 'operation_mode'] = \
                    translation['comp'][language]
                # assumption that consumption < 100 W is pumps only
                profile.loc[profile['P_TOT'] < 100, 'operation_mode'] = \
                    translation['pumps'][language]
                profile['operation_mode'].fillna(
                    translation['rod'][language], inplace=True)
            else:
                profile.loc[:, 'operation_mode'] = translation['hh'][language]
            profile = profile.groupby('operation_mode').sum()
            profile.loc['obj'] = obj
            total = pd.concat([total, profile['P_TOT']], axis=1)
        missing = pd.DataFrame.from_dict(missing).stack().swaplevel(0, 1)
        miss_for_merge = missing.groupby(level=0).sum()
        miss_for_merge.name = 'missing'
        total = total.T.reset_index(drop=True)
        total = total.set_index('obj')
        total = total.merge(miss_for_merge, left_index=True, right_index=True)
        # merge household and heat pump rows
        total = total.astype(
            {v[language]: 'float64' for k, v in translation.items()})
        total = total.groupby(level=0).sum()
        total['nr'] = total.index.str.split('(\d+)').str[1].astype(int)
        total['building_nr'] = total.index
        total = total.sort_values(by='nr', ascending=True).set_index(
            'building_nr')
        total = total.drop('nr', axis=1)
        cols = [val[language] for key, val in translation.items()]
        total[cols] = total[cols] / 360 / 1e3
        # plot
        fig, ax = plt.subplots(figsize=(16, 9))
        # heat pump
        colors = ['gainsboro', 'yellowgreen', 'orangered']
        cols = [val[language] for key, val in translation.items()
                if key != 'hh']
        total.loc[total['missing'] == 4, cols].plot(
            ax=ax, kind='bar', stacked=True, color=colors, position=1,
            width=0.4
        )
        # household
        total.loc[total['missing'] == 4, translation['hh'][language]].plot(
            ax=ax, kind='bar', color='cadetblue', position=0,
            width=0.4, edgecolor='white', linewidth=1
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='best')
        if language == 'de':
            ax.set_ylabel('Wirkenergie [kWh/a]')
        elif language == 'en':
            ax.set_ylabel('Annual active energy consumption in kWh/a')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=30)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
        ax.set_xlim(left=ax.get_xlim()[0] - 0.25)
        if strfile:
            if os.path.isdir(strfile):
                strfile = os.path.join(
                    strfile, 'heat_pump_operation_wpuq_report.png')
            if os.path.isfile(strfile):
                os.remove(strfile)
            plt.savefig(strfile, bbox_inches='tight', dpi=300)
            total.to_csv(strfile.replace('.png', '.csv'))
            plt.close()
