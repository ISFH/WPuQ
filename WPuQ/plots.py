from datetime import datetime as dt
from calendar import month_abbr
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
                energy[direct] = (pd.read_hdf(source_file_energy, dset_name)
                                  [col_name])
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
        combined = pd.read_hdf(source_file, dset_name, columns=cols_c).sum(
            axis=1)
        # individual measurements of I and U
        if phase == 'TOT':
            cols_i = [c + '_' + str(p) for c in translation['individual']
                      for p in [1, 2, 3]]
        else:
            cols_i = [c + '_' + phase for c in translation['individual']]
        indiv = pd.read_hdf(source_file, dset_name, columns=cols_i)
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


    def plot_seasonal_load_curves(self, folder, objects, feed, res,
                                  correct_pv=True, plot_slp=False,
                                  use_average=False, figtitle=False):
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
            except KeyError:
                continue
            print(dt.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Plotting ' +
                  f'seasonal load curves for dset {dset_name}.')
            if correct_pv or 'P_TOT_WITH_PV' not in profile.columns:
                profile = profile['P_TOT']
            else:
                profile = profile[['P_TOT', 'P_TOT_WITH_PV']]
            if use_average:
                try:
                    n_objs = len(
                        file[dset_name + '/table'].attrs['objects_included'])
                    profile = profile / n_objs
                except KeyError:
                    pass
            profile.index = profile.index.tz_localize(
                'Europe/Berlin', nonexistent='shift_forward', ambiguous='NaT')
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
                slp = e_slp.get_profile(
                    {slp_name: profile.sum() / 360 * 4})[slp_name]
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
