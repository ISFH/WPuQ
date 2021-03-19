import setuptools
from os import path

dir = path.abspath(path.dirname(__file__))
with open(path.join(dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='WPuQ',
    version='1.0',
    description='Software to download, process and plot WPuQ data',
    maintainer='Marlon Schlemminger',
    maintainer_email='m.schlemminger@isfh.de',
    url='',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=['pandas', 'numpy', 'h5py', 'matplotlib', 'tables', 'pytz'],
    package_data={
        'WPuQ': ['data/*.csv']
    },
    long_descpription=long_description,
    long_description_content_type='test/markdown'
)
