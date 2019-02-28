# New York City Green Taxi Data Analysis

This project used NYC green taxi data collected by the NYC Taxi and Limousine Commission. This folder contains code and report of my analysis of the September 2015 data. For a further dive into the green taxi background and data set, see [NYC TLC](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml). 

## Folder Structure:
- `Report.pdf`: Full report for the entire project
- `Analysis.ipynb`: Jupyter notebook for project workflow visualization. 
- `functions.py`: Helper functions used in the above Jupyter notebook
- `environment.yml`: This file includes environment requirements and necessary python packages for this project. Need to be installed using Anaconda (i.e. conda env create -f environment.yml).

## Quick start:
This section has the commands to quickly get started running code in this folder.
For more detailed installation instructions, see the `Python Environment` below.
These instructions assume that you already have [conda](https://conda.io/) installed.

First, navigate to the root of the `nyc_green_taxi-C918025` directory in a terminal and run the following:

```bash
# Install the environment
conda env create --file=environment.yml

# Activate the environment
source activate nyc_green_taxi

# Initiate a jupyter notebook server
jupyter notebook
```
Then a Jupyter notebook tab will open in your browser. From here you can open and run the existing Analysis.ipynb from beginning


### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use environment from file*, and *Change environments (activate/deactivate)*). 
