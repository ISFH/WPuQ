import os
import datetime
import json

import h5py

from WPuQ.h5ls import H5ls


def get_hdf5_resources(filename):
    '''
    Creates the metadata of all data resources of an HDF5 file

    Parameters
    ----------
    filename : str
        The filename

    Returns
    -------
    resources : dict
        The metadata

    '''
    file = h5py.File(filename, 'r')
    visitor = H5ls()
    file.visititems(visitor)
    dset_names = visitor.names
    name = os.path.split(filename)[-1].split('.')[0]
    path = os.path.split(filename)[-1]
    profile = 'data-resource'
    fields = list()
    units = dict(
        weather={
            'APPARENT_TEMPERATURE_TOTAL': '°C',
            'ATMOSPHERIC_PRESSURE_TOTAL': 'mbar',
            'PRECIPITATION_RATE_TOTAL': 'mm',
            'PROBABILITY_OF_PRECIPITATION_TOTAL': '%',
            'RELATIVE_HUMIDITY_TOTAL': '%',
            'SOLAR_IRRADIANCE_GLOBAL': 'W/m2',
            'TEMPERATURE_TOTAL': '°C',
            'WIND_DIRECTION_TOTAL': '°',
            'WIND_GUST_SPEED_TOTAL': 'm/s',
            'WIND_SPEED_TOTAL': 'm/s'
        },
        district_heating={
            'HEAT_TEMPERATURE_FLOW': '°C',
            'HEAT_TEMPERATURE_RETURN': '°C'
        }
    )
    for dset_name in dset_names:
        dset_name = dset_name.replace('/table', '')
        if 'spatial' in filename:
            try:
                loc, res, feed = dset_name.split('/')
                description = (
                    f'Cumulated load of active and reactive power over all '
                    f'objects classified as {loc} in the temporal resolution '
                    f'of {res} for the feed {feed}.'
                )
            # substation has less layers
            except ValueError:
                loc, res = dset_name.split('/')
                description = (
                    'Cumulated load of active and reactive power of the object'
                    f' {loc} in the temporal resolution of {res}.'
                )
            unit = dict(
                P='W',
                Q='VAR'
            )
        elif 'weather' in filename:
            service, direct, param_long = dset_name.split('/')
            param = param_long.split('_', 1)[1]
            description = (f'Timeseries of the weather parameter {param}.')
            unit = units['weather'][param]
        elif 'district_heating' in filename:
            service, direct, param = dset_name.split('/')
            description = ('Timeseries in the district heating grid of '
                           f' parameter {param}.')
            unit = units['district_heating'][param]
        else:
            try:
                loc, obj, feed = dset_name.split('/')
            # PV has more layers
            except ValueError:
                loc, obj, feed1, feed2, feed3 = dset_name.split('/')
                feed = '_'.join([feed1, feed2, feed3])
            description = (
                f'Electrical properties of the feed {feed} in object {obj} '
                f'classified as {loc}.'
            )
            unit = dict(
                S='VA',
                P='W',
                Q='VAR',
                PF='no unit',
                U='V',
                I='A'
            )
        description += ' The index is the unix timestamp in nanoseconds. '
        field = dict(
            name=dset_name,
            type='number',
            description=description,
            unit=unit,
            )
        fields.append(field)
    schema = dict(
        fields=fields)
    resources = dict(
        name=name,
        profile=profile,
        path=path,
        schema=schema
    )
    return resources


def create_metadata(folder):
    '''
    Creates a json-file containing metadata of the full datapackage

    Parameters
    ----------
    folder : str
        The folder storing the HDF5 files

    Returns
    -------

    '''
    resources = []
    for filename in os.listdir(folder):
        if not (filename.startswith(('data', 'weather', 'district_'))
                and filename.endswith('.hdf5')):
            continue
        resources.append(get_hdf5_resources(os.path.join(folder, filename)))
    name = 'WPuQ household and heat pump electric load profiles'
    idd = r'10.5281/zenodo.4719836'
    licenses = list(
        [dict(
            name='CC-BY-4.0',
            title='Creative Commons Attribution 4.0',
            path='https://creativecommons.org/licenses/by/4.0/'
        )]
    )
    profile = 'data-package'
    description = (
        'Electric load of 38 households measured in a small village in Lower '
        'Saxony, Germany. Data is available for voltage, reactive power and '
        'active power. Seperate measurements are available for each total '
        'household load, for each heat pump and at the local power '
        'transformer. Different temporal and spatial aggregations are '
        'available for conveniance.'
    )
    homepage = r'10.5281/zenodo.4719836'
    version = '1.0'
    contributors = list(
        [dict(
            title='Marlon Schlemminger',
            email='m.schlemminger@isfh.de',
            role='author',
            organization='Institute for Solar Energy Research in Hamelin'
        ),
        dict(
            title='Tobias Ohrdes',
            email='ohrdes@isfh.de',
            role='author',
            organization='Institute for Solar Energy Research in Hamelin'
        ),
        dict(
            title='Elisabeth Schneider',
            role='contributor',
            organization='Institute for Solar Energy Research in Hamelin'
        ),
        dict(
            title='Michael Knoop',
            role='contributor',
            organization='Institute for Solar Energy Research in Hamelin'
            )]
    )
    documentation = 'URL to paper'
    spatial = dict(
        location='Lower Saxony, Germany',
        resolution='single-family houses'
    )
    temporal = dict(
        start='01-05-2018',
        end='31-12-2020',
        resolution='10s'
    )
    keywords = list(['electricity consumption', 'household load profile',
                     'buildings', 'heat pump load profile',
                     'quarter', 'open data'])
    created = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    metadata = dict(
        profile=profile,
        name=name,
        contributors=contributors,
        id=idd,
        description=description,
        homepage=homepage,
        documentation=documentation,
        spatial=spatial,
        temporal=temporal,
        version=version,
        licenses=licenses,
        keywords=keywords,
        created=created,
        resources=resources
    )
    json_filename = os.path.join(folder, 'datapackage.json')
    with open(json_filename, 'w') as fp:
        json.dump(metadata, fp)


if __name__ == '__main__':
    quarter = 'Ohrberg'
    years = [2018, 2019, 2020]
    for year in years:
        folder = os.path.join(os.getcwd(), 'data_to_publish', f'{year}')
        create_metadata(folder)
