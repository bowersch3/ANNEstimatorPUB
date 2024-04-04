# ANN_Estimator_PUB **(Under Peer Review)**
The python file used to build the feedforward neural network (FNN) and generate the figures from the manuscript.

This repository is released for the peer review process and has NOT been accepted.

## Contents
- [About](#about)
- [Pre-processing](#pre-processing)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About
The Pub_ANN_Work.py contains functions that was used to build and train a FNN to estimate the upstream interplanetary magnetic field (IMF) conditions from magnetosheath measurements obtained by the MESSENGER spacecraft. This model is useful for decreasing the temporal separation between an estimate of the IMF and a magnetospheric phenomenon measured by MESSENGER by inferring IMF conditions based on magnetosheath measurements.

The model makes IMF predictions for all magnetosheath measurements taken by MESSENGER from March 24, 2011 to April, 29, 2015. The time ranges for the magnetosheath are determined by a list of crossings compiled by Weijie Sun and available via a Zenodo repository (https://zenodo.org/record/8298647).

The MESSENGER data must be downloaded via the Planetary Plasma Interactions Node of the Planetary Data System (https://pds-ppi.igpp.ucla.edu/). This study utlized the 1 second resolution magnetometer dataset (MESSENGER MAG Time-Averaged Calibrated MSO Coordinates Science Data Collection).

## MESSENGER Data Format
The functions within PUB_ANN_Work.py assume that the entirety of the MESSENGER MAG Time-Averaged Calibrated MSO Coordinates Science Data Collection has already been downloaded.

The data are organized such that data from each calendar year is in a separate folder, which is divided into folders of numbered months, and all data within that month is contained within the same folder.

The path to the MESSENGER dataset must be included at the beginning of the code under the variable: 

```python
file_MESSENGER_data = "YOUR_MESSENGER_DATA"
```

The following format is required in order to generate dataframes necessary for this analysis:

```python
file_MESSENGER_data = 'path_to_MESSENGER_data/mess-mag-calibrated - avg'+'year'+'/'+month+'/'+'MAGMSOSCIAVG'+year+doy+'_01_V08.TAB'
```
Here, 'year' is the last two digits of the calendar year (i.e., 11,12,13...), 'month' is the number of the calendar month (01,02,03 etc), and 'doy' is the day of the year.

Example:

'/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated - avg11/05/MAGMSOSCIAVG11121_01_V08.TAB'



## Usage
Show examples and explain how to use your project. Provide code snippets if necessary.

## Contributing
Include guidelines for contributing to your project, such as how to report bugs or suggest improvements. 

## License
Specify the license under which your project is distributed. For example, you can use the MIT License, Apache License 2.0, etc. Provide a link to the license file if applicable.
