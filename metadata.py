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
    for dset_name in dset_names:
        dset_name = dset_name.replace('/table', '')
        if 'spatial' in filename:
            loc, res, feed = dset_name.split('/')
            description = (
                f'Cumulated load of active and reactive power over all objects'
                f' classified as {loc} in the temporal resolution of {res} '
                'for the feed {feed}.'
            )
            unit = dict(
                P='W',
                Q='VAR'
            )
        else:
            loc, obj, feed = dset_name.split('/')
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
        description += 'The index is the unix timestamp in nanoseconds. '
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
        if not filename.startswith('data' and filename.endswith('.hdf5')):
            continue
        resources.append(get_hdf5_resources(filename))
    name = 'WPuQ household and heat pump electric load profiles'
    idd = 'Daten DOI'
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
    homepage = 'Link zum Daten repository'
    version = '1.0'
    contributors = list(
        [dict(
            title='Marlon Schlemminger',
            email='schlemminger@solar.uni-hannover.de',
            role='author',
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
    json_filename = os.path.join(os.path.split(filename)[0],
                                 'datapackage.json')
    with open(json_filename, 'w') as fp:
        json.dump(metadata, fp)


if __name__ == '__main__':
    quarter = 'Ohrberg'
    years = [2018, 2019, 2020]
    for year in years:
        folder = os.path.join(os.getcwd(), f'{quarter}_{year}', 'resampled')
        create_metadata(folder)
