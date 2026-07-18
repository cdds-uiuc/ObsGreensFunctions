# Observational Greens Function - design document


**Goal:** The goal of this project is to asses variations in net radiative feedbacks over historical simulations using observationally-derived greens functions. 

**Principle**
- you can start from the code in the current directory to get some context, but the current codebase needs to be almost entirely refactored.
- Simplicity is a key principle here. We want a fairly simple codebase, that is not overengineered. This is a fairly small and self-constrained project, so we do not needt o add in too much flexibility.

**Structure:**
 - **raw data:** use the raw data in the local cmip6 holdings here: /Users/cristi/cmip6/ . details on using this intake-esm based catalog can be found in /Users/cristi/cmip6/catalog/USING_THE_CATALOG.md. 
 - **Pre-processing**:  we want a separate pre-processing step that loads the data and output pre-processed data.
     - Pre-processing should include harmonizing the data (detailed in USING_THE_CATALOG.md).
     - we only need net top-of-atmosphere radiation, created by combining rsut,rlut and rsdt (with appropriate signs) intto a new variable called "toa".
     - we only need to keep annual-means of the data.
     - the pre-processed data should consist of netcdf files with the following rules:
         - one file per variable, per experiment, per model, per ensemble. The raw data for each experiment is sometimes split in many files with a small number of years each. combine these into a single file.
         - These files should be annual means.
         - The processed files should use cmip6 naming conventions, e.g.: tas\_Amon\_CanESM5\_historical\_r23i1p1f1\_gn\_1850-2014.nc  and toa\_Amon\_CanESM5\_historical\_r23i1p1f1\_gn\_1850-2014.nc , except for the taking out the months (since we're using annual means)
         - the three variables should be tas,toa, and tos (for the experiments that report tos).
         - the pre-processed data should live in a "pre-processed_data" folder
         - for this project, we want to keep each model on its native grid. 
 **Analysis**
- we want to do some of the following analysis:
- for amip-piForcing simulations, we want to compute a radiative feedback using 30-year windows, by regressing net toa anomalies against net tas anomalies in these 30-year windows (anomalies should be defined relative to the first 50 years of the amip-piForcing simulation)
- we then want to build observational greens functions for \partial tas_global / \partial tas(x) and \partial toa_global / \partial tas(x). We want to use SSTs, but since these are not reported, we can just use tas over the oceans (we may need another file to determien where there is land and where there is ocean, let me know which file that is and I can download it)
- we then want to compare the 30-year wndow feedbacks obtained directly from amip-piForcing toa and tas with the 30-year window feedbacks obtained from greens function reconstructions. 
- we will then take tos data from historical simulations, and compute the same feedbacks using greens function reconstructions of tas and toa based on applying the amip-piForcing derived greens functions to tos data from coupled historical simulations.
- finally we want a plot of the time-series of the feedbacks in the amip-piForcing simulation (from model, and GF-derived) and the time-series of feedbacks from historical simulations (GF-derived)
        


       
 

