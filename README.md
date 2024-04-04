# ANN_Estimator_PUB **(Under Peer Review)**
The python file used to build the feedforward neural network (FNN) and generate the figures from the manuscript.

This code trains and tests an FNN to predict upstream IMF conditions based on magnetosheath observations obtained by MESSENGER. It it's current form, the model must only be applied to MESSENGER data, and is not directly applicable to other spacecraft.

This repository is released for the peer review process and has NOT been accepted.

## Contents
- [About](#about)
- [Data](#data)
- [Usage](#usage)
- [Model](#model)
- [Assessment](#assessment)
- [Limitations](#limitations)
- [License](#license)

## About
The Pub_ANN_Work.py contains functions that were used to build and train a FNN to estimate the upstream interplanetary magnetic field (IMF) conditions from magnetosheath measurements obtained by the MESSENGER spacecraft. This model is useful for decreasing the temporal separation between an estimate of the IMF and a magnetospheric phenomenon measured by MESSENGER by inferring IMF conditions based on magnetosheath measurements.

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
First, at the beginning of the PUB_ANN_Work.py, specify a folder where the data and results may be saved under the variable:
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
- The model is trained on 8 features, and outputs 4 targets
  - Split the data into training, test, and validation datasets.
  - Normalize the inputs and outputs using MinMaxScaler from sklearn.preprocessing
  - Randomly pull data from the training set using bootstrap aggregation
  - The model:
    - Input layer
    - 3 hidden layers with 24 neurons per layer and a hyperbolic tangent activation function and a BatchNormalization layer in between each hidden layer
    - Output layer
  - The model is then saved and a new model is built and trained with slightly different training data from the next bootstrap aggregation pull of the training data
- Predictions for the IMF for MESSENGER magnetosheath observations are made by running all $n$ model predictions on all magnetosheath measurements, taking the average and standard deviation of all model predictions for each magnetosheath measurement.
- These predictions are then saved to a .pkl file.

## Model

The model uses a feed-forward neural network. The coordinate system for the model is the aberrated Mercury Solar Magnetospheric (MSM') coordinate system. This coordinate system is centered on the magnetic dipole of Mercury, where $\hat{X}$ points in the opposite direction of the solar wind flow, $\hat{Y}$ is opposite to the orbital motion of Mercury, and $\hat{Z}$ points towards geographic north. The inputs to the model are measurements taken within the magnetosheath. The inputs to the model are:

- Magnetosheath $B_X$, $B_Y$ and $B_Z$ (nT)
- Amplitude of the magnetosheath magnetic field $|B|$ (nT)
- $X$ position of MESSENGER ($R_M$) where $R_M$ is the radius of Mercury (2440 km)
- $r$, where $r$ is the cylindrical radius ($r = \sqrt{Y^2+Z^2}$) ($R_M$)
- $\Theta$ where $\Theta$ = $\arctan(\frac{Y}{Z})$
- Heliocentric distance of Mercury (AU)

The outputs of the model are:

- IMF $B_X$, $B_Y$ and $B_Z$ (nT)
- $|B|$ of the IMF (nT)

## Assessment

- The $r^2$ score of the model was assessed for the test set for each output indivdually:
  - IMF $B_X$: $r^2 = 0.81$
  - IMF $B_Y$: $r^2 = 0.66$
  - IMF $B_Z$: $r^2 = 0.58$
  - IMF $|B|$: $r^2 = 0.76$
- Average $r^2$ for all outputs $= 0.70$

## Limitations
- This model is trained on data obtained for the entirety of MESSENGER data. Therefore, the model in it's current form should not be applied to magnetosheath observations of other planetary environments or other spacecraft observations of Mercury
- Because IMF $|B|$ is predicted independently of its components, the predicted IMF components will not add in quadrature to match the IMF $|B|$ prediction, care should be taken when comparing components of the IMF prediction to the IMF $|B|$ prediction.
- When any of magnetosheath magnetic field components are low, the model tends to perform worse.
- The model only makes predictions at a 40 s cadence and should not be trained on or applied to a higher resolution measurements. This constraint is due to the high frequency variability of magnetosheath magnetic field that are caused by local processes rather than upstream IMF rotations.
- 
## License
Specify the license under which your project is distributed. For example, you can use the MIT License, Apache License 2.0, etc. Provide a link to the license file if applicable.
