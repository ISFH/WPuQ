# WPuQ
*  Author: Marlon Schlemminger
*  Contact: m.schlemminger@isfh.de
*  Version: 1.0

## Description
This package includes the code used to download, process, validate and visualize the measurement data of the WPuQ project. 

## Install instructions
Clone this repository to a folder of your choice:
```git clone https://github.com/ISFH/WPuQ```

Create a new conda environment:
```conda create --name wpuq python=3.7```

Activate the environment:
```conda activate wpuq```

Install the package with all requirements:
```pip install .```


Run the executable script by:
```python appl.py```

Downloading the original data requires an API key, which we unfortunately can not provide to outside persons due to privacy reasons. Therefore, certain sections in the appl.py are commented out, because they are not executable without an API key. We still include these sections in the script to show how the download works in theory. The remaining code requires you to download the processed data from [here]() and set the appropriate path to the data in appl.py.
 

## Documentation
A scientific documentation of the dataset can be found under this DOI:
