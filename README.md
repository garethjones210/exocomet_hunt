# exocomet_hunt

Code for automated detection of comets in light curve files.

### Installation

Requires Python 3 (tested in 3.5). Uses Numpy, Scipy, Astropy and Matplotlib libraries, and a working Cython install. 

Install by running:

    git clone https://github.com/garethjones210/exocomet_hunt
    cd exocomet_hunt
    ./make

### Usage

These scripts runs on light curve files, which can be obtained from [MAST](https://archive.stsci.edu/kepler/).

`single_analysis.py` runs on a single file, for example:

    wget https://archive.stsci.edu/missions/kepler/lightcurves/0035/003542116/kplr003542116-2012088054726_llc.fits
    ./single_analysis.py kplr003542116-2012088054726_llc.fits

`batch_analyse.py` runs on directories of files, outputting results to a text file with one row per file. `archive_analyse.sh` is a bash script for processing compressed archives of light curve files, extracting them temporarily to a directory.  Both these scripts have multiple options (number of threads, output file location ...), run with help flag (`-h`) for more details.

### Output

https://github.com/garethjones210/exocomet_hunt_TESS_sector_results contains a description of the output table format produced by this code, as well as the output when run on the TESS datasets from sectors 1 through 11. Note that this does not include any full frame images (FFIs). See there also for description of the format of the txt files with dips, and how to filter then with the awk scripts.

### Other files

* The jupyer notebook figs.ipynb contains code to explore individual light curves, and makes many different plots.
