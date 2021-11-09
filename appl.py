import os
from datetime import datetime as dt

from WPuQ import Datacollector, Dataprocessor, WPuQPlots

if __name__ == '__main__':
    corrections_tot = dict(
        Ohrberg=dict(
            datacollector=dict(timestamps=False,
                               device_bounds=False,
                               outliers=False
                               ),
            dataprocessor=dict(timestamps=True,
                               device_bounds=True,
                               outliers=False
                               )
            ),
        Huegelshart=dict(
            datacollector=dict(timestamps=False,
                               device_bounds=False,
                               outliers=False
                               ),
            dataprocessor=dict(timestamps=True,
                               device_bounds=False,
                               outliers=False
                               )
            )
        )
    years = [2018, 2019, 2020]
    quarter = 'Ohrberg'
    folder_base = 'F:\schlemminger\WPuQ\Python_Implementation\WPuQ'
    for year in years:
        print(dt.now().strftime('%m/%d/%Y, %H:%M:%S') + ' WPuQ dataprocessing'
              + f' for year {year}.')
        folder = os.path.join(folder_base, f'{quarter}_{year}')
        # downloading data only works with an apikey
        # corrections = corrections_tot[quarter]['datacollector']
        # collector = Datacollector()
        # collector.get_objects(quarter=quarter)
        # collector.get_data(start_month=f'01-{year}', end_month=f'12-{year}',
        #                     time_interval=10, corrections=corrections,
        #                     weather_data=False)
        # restructure data
        corrections = corrections_tot[quarter]['dataprocessor']
        processor = Dataprocessor()
        processor.get_pv_objects(quarter=quarter)
        processor.aggregate_temporal(folder=folder, corrections=corrections)
        if quarter != 'Ohrberg':
            continue
        processor.prove_consistency(folder=folder, corrections=corrections)
        processor.detect_heating_rod_operation(
            folder=folder, strfile=os.path.join(folder, 'validation'))
        processor.aggregate_spatial(folder=folder)
        # plot data
        plotter = WPuQPlots()
        plotter.get_pv_objects(quarter=quarter)
        plotter.plot_seasonal_load_curves(
            folder=os.path.join(folder, 'resampled'), objects=None,
            feed='HOUSEHOLD', res='10s', correct_pv=False, use_average=True,
            plot_slp=True)
        plotter.plot_seasonal_load_curves(
            folder=os.path.join(folder, 'resampled'), objects=None,
            feed='HEATPUMP', res='10s', correct_pv=False, use_average=True,
            plot_slp=True)
        plotter.plot_annual_consumption(folder=folder, strfile=folder)
        plotter.plot_annual_consumption_household_for_report(
            folder=folder,
            strfile=os.path.join(
                folder_base,
                f'household_operation_wpuq_report_{year}.png'
            )
        )
        method = 'larger 4kW'
        plotter.plot_annual_consumption_heat_pump_for_report(
            folder=os.path.join(folder, 'resampled'), method=method,
            language='en',
            strfile=os.path.join(
                folder_base,
                f'heat_pump_operation_wpuq_report_{year}_{method}.png'
            )
        )
        for language in ['de', 'en']:
            if language == 'de':
                ymax = 255
            else:
                ymax = None
            plotter.plot_daily_for_report(
                os.path.join(folder, 'resampled'), language, ymax,
                strfile=os.path.join(
                    folder_base, f'daily_plot_wpuq_report_{year}_{language}.png')
            )
    # plots for all years
    if quarter == 'Ohrberg':
        plotter = WPuQPlots()
        for power in ['POWER', 'ENERGY']:
            for feed in ['HOUSEHOLD', 'HEATPUMP']:
                plotter.plot_data_quality(
                    folder=folder_base, quarter=quarter, years=years,
                    feed=feed, power=power, strfile=folder_base
                )
