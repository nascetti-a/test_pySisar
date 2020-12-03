# pyDATE - (Digital Automatic Terrain Extractor)

pyDATE (Digital Automatic Terrain Extractor) is a new software package able to generate Digital Surface Models (DSMs) from high resolution optical and SAR satellite imagery acquired by the most common sensors.

# Installation and dependencies

The main source code repository for this project is https://github.com/nascetti-a/test_pySisar 
It is written in Python. It was tested with Python 3.5, 3.6 in Linux and MacOS-X operating systems. 

`pyDATE` requires `libtiff` development files. They can be installed with `apt-get install libtiff-dev` (Ubuntu, Debian) or `brew install libtiff` (macOS)

Once `libtiff` is installed, `pyDATE` can be install using Conda. Please use the following steps:

Create a new conda environment using the env.yml file 

    `conda env create -f envshort.yml ` 
    
# Usage 

To run `pyDate` please enter in the bin folder and run the program:

    python pyDate.py config.json

An example of a configuration file is available in the bin folder (I will add a full description of all the possible parameters)

# References 

1. Martina Di Rita, Andrea Nascetti & Mattia Crespi (2017) Open source tool for DSMs generation from high resolution optical satellite imagery: development and testing of an OSSIM plug-in, International Journal of Remote Sensing, 38:7, 1788-1808, DOI: 10.1080/01431161.2017.1288305

2. Martina Di Rita, Andrea Nascetti & Mattia Crespi (2018) FOSS4G DATE for DSMs generation from tri-stereo optical satellite images: development and first results, European Journal of Remote Sensing, 51:1, 472-485, DOI: 10.1080/22797254.2018.1450644


    

