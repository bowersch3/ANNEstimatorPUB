# ANN_Estimator_PUB **(Under Peer Review)**
The python file used to build the feedforward neural network (FNN) and generate the figures from the manuscript.

This repository is released for the peer review process and has NOT been accepted.

## Contents
- [About](#about)
- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About
The Pub_ANN_Work.py contains functions that was used to build and train a FNN to estimate the upstream interplanetary magnetic field (IMF) conditions from magnetosheath measurements obtained by the MESSENGER spacecraft. This model is useful for decreasing the temporal separation between an estimate of the IMF and a magnetospheric phenomenon measured by MESSENGER by inferring IMF conditions based on magnetosheath measurements.

The model makes IMF predictions for all magnetosheath measurements taken by MESSENGER from March 24, 2011 to April, 29, 2015. The time ranges for the magnetosheath are determined by a list of crossings compiled by Weijie Sun and available via a Zenodo repository (https://zenodo.org/record/8298647).

The MESSENGER data must be downloaded via the Planetary Plasma Interactions Node of the Planetary Data System (https://pds-ppi.igpp.ucla.edu/). This study utlized the 1 second resolution magnetometer dataset (MESSENGER MAG Time-Averaged Calibrated MSO Coordinates Science Data Collection).

## Data
The functions within PUB_ANN_Work.py assume that the entirety of the MESSENGER MAG Time-Averaged Calibrated MSO Coordinates Science Data Collection has already been downloaded.

The data are organized such that data from each calendar year is in a separate folder, which is divided into folders of numbered months, and all data within that month is contained within the same folder.

The path to the MESSENGER dataset must be included at the beginning of the code in the PUB_ANN_Work.py under the variable: 

```python
file_MESSENGER_data = "YOUR_MESSENGER_DATA"
```

The following format is required in order to generate dataframes necessary for this analysis:

```python
file_MESSENGER_data = 'path_to_MESSENGER_data/mess-mag-calibrated - avg'
```
Then, year, month and day of year are added in the filename upon loading. An example of a single day's worth of magnetic field data should look like:

'path_to_MESSENGER_data/mess-mag-calibrated - avg'+'year'+'/'+month+'/'+'MAGMSOSCIAVG'+year+doy+'_01_V08.TAB'

Here, 'year' is the last two digits of the calendar year (i.e., 11,12,13...), 'month' is the number of the calendar month (01,02,03 etc), and 'doy' is the day of the year.

Example:

'/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated - avg11/05/MAGMSOSCIAVG11121_01_V08.TAB'



## Usage
First, at the beginning of the PUB_ANN_Work.py, specify a folder where the data and results may be saved under the variale:
```python
save_path = 'YOUR_PATH_TO_FOLDER'
```
Then, specify where the Sun2023 (https://zenodo.org/record/8298647) list of boundary crossings (saved as .txt files) are downloaded under the variables:

```python
file_mp_in = 'PATH_TO_MAG_PAUSE_IN'
file_mp_out = 'PATH_TO_MAG_PAUSE_OUT'
file_bs_in = 'PATH_TO_BOW_SHOCK_IN'
file_bs_out = 'PATH_TO_BOW_SHOCK_OUT'
```
- From the 1 sec resolution magnetic field data downloaded from the PDS and saved in the format above, we generate a dataframe that contains all magnetic field data observed by MESSENGER, and then pick-out all magnetosheath traversals based on the Sun2023 list of boundary crossings.
- The dataset of all magnetosheath measurements is then downsampled to 40 second resolution
- We generate the dataframes that become magnetosheath inputs and outputs by selecting for bow shock crossing intervals that meet the conditions outlined by the publication.
- This dataframe of features (magnetosheath properties) and targets (IMF properties) is then used to train a total of $n$ feed-forward neural networks, where $n$ is the number of models used to build an ensemble model trained on slighlty different training data.
  - Split the data into training, test, and validation datasets.
  - Randomly pull data from the training set using bootstrap aggregation

## Contributing
Include guidelines for contributing to your project, such as how to report bugs or suggest improvements. 

## License
Specify the license under which your project is distributed. For example, you can use the MIT License, Apache License 2.0, etc. Provide a link to the license file if applicable.
