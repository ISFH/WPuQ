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
    for year in years:
        print(dt.now().strftime('%m/%d/%Y, %H:%M:%S') + ' WPuQ dataprocessing'
              + f' for year {year}.')
        folder = os.path.join(os.getcwd(), f'{quarter}_{year}')
        # downloading data only works with an apikey
        # corrections = corrections_tot[quarter]['datacollector']
        # collector = Datacollector()
        # collector.get_objects(quarter=quarter)
        # collector.get_data(start_month=f'01-{year}', end_month=f'12-{year}',
        #                    time_interval=10, corrections=corrections,
        #                    weather_data=False)
        # restructure data
        corrections = corrections_tot[quarter]['dataprocessor']
        processor = Dataprocessor()
        processor.get_pv_objects(quarter=quarter)
        # processor.aggregate_temporal(folder=folder, corrections=corrections)
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

    # plots for all years
    if quarter == 'Ohrberg':
        plotter = WPuQPlots()
        for power in ['POWER', 'ENERGY']:
            for feed in ['HOUSEHOLD', 'HEATPUMP']:
                plotter.plot_data_quality(
                    folder=os.getcwd(), quarter=quarter,
                    years=years,
                    feed=feed, power=power, strfile=os.getcwd()
                )
