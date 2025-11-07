# LEDMSI Readme

***

## Introduction
This document is a is a readme file for the LEDMSI system and analysis software contained within this directory. It has 3 main functions:
 - System control and calibration
 - Data processing and analyses
 - Plotting the analyses for the paper

The structure of this document is:
  - A description of the directory structure
  - A description of the software environment and how to run it
  - An explanation of the system control logic
  - A description of the image data structure
  - A description of the analysis methods
  - A description of the how to derive the analyses discussed in the paper
  
A test program is included that does 

***
  
## Directory structure
This describes the directory stucture:

/titan_LEDMSI_data
 - /code/
	  - /dockerfile * the docker environment for running the code*
	  - /python/ *contains the Python code and associated config files*
		- /bandnames.txt *which channels map to which bands*
		- /board_control.py *python code that controls the arduino*
		- /CA_plots.py *component analysis plots (ICA and PCA*
		- /DataHandler.py *Data structure classes with methods that set up a dicts that point to images and associate them with spectral data, json import and export for these dicts.*
		- /DngFluo.py *fluorescence from raw files*
		- /example.py *examples of how to recreate the analyses shown in the paper*
		- /jpeg_fluo.py
		- /lda.py *performs and plots plots linear discriminant analysis* 
		- /led_spectra_camera.py *plots spectra for LEDs and camera sensitivity*
		- /processing.py *Classes that handle thce oconversion of the iages to Numpy arrays and their subsequent processing, visualisation and analyses
		- /requiremnts_conda.txt
		- /requirements.txt *requirmennts for Python dependencies*
		- /SysController.py *Controls the system from the command line, including image aquisition and calibration*
		- /SysHandler.py 
		- /system_definition.json *json file that maps the Arduino pins to LED bands*
 - /data/
	 - /images/
		 - /full_images/
			- /watts_no_filter_2/
			- /watts_no_filter_6/
		 - /filtered/
			 - 400_filter_2 *image aquired with 400nm longpass filter*
		 - /masks *images for sampling for LDA analysis*
		 - /samples
			 - Titan_samples.json *samples used for the LDA analyses*
		 - /subsets/ *subsets for recreating the reslts in the paper*
	 - /led_spectra *measured emission spectra for LEDs* 
	 - /sensor_sensitivity *Measured spectral sensitivity of the camera sensor with hot mirror filter removed*


## System control
Without the hardware this stage is difficult to replicate. How it works:
 - Using the system definition json file the system turns on the led for each wavelength and takes a photo
 - It saves the photo in a directory generated from the command line input
  - saves which photo correscponds to which wavelength in a json file in that directory. These are then used to order the bands when stacking the multispectral imag

## Software environment
The included dockerfile bundles all dependencies with the assosciated data in this directory structure. Examples of how to use this to run the analyses code are below 

### Build the docker image:
From this directory:

    docker build -t ledmsi_dockerfile .
(dont forget the trailing period!)

### Run the image in a container
We'll run this by **bind-mounting** the image so we can access the data in the container on the host machine:

In bash(ish):

	docker run -it -v $(pwd)/code:/titan_LEDMSI_data/code -v $(pwd)/data:/titan_LEDMSI_data/data -v $(pwd)/output:/titan_LEDMSI_data/output -w /titan_LEDMSI_data ledmsi_dockerfile
 
In windows powershell:

	docker run -it -v ${PWD}/code:/titan_LEDMSI_data/code -v ${PWD}/data:/titan_LEDMSI_data/data -v ${PWD}/output:/titan_LEDMSI_data/output -w /titan_LEDMSI_data ledmsi_dockerfile


## Run the included example scripts
First cd to the right diretory:

	cd code/python
	
To plot the LED spectra:

	python led_spectra_camera.py
	
This will plot the led spectra and camera sensitivity used in the paper. The plots will appear in the output directory 

To perform the PCA, ICA analyses discussed in the paper:

	python example.py folder
 
will re-read the raw data and perform the analyses. The plots will be in the outputs folder. The plots presented in the paper are cropped slightly to remove the influence of vignetting in the source images. To replicate these exactly run the following:

	python example.py paper
	
Finally, to run the hue-difference flourescence run:

	python example.py fluo
	
To run the LDA analysis run the lda.py script. This uses processing.SampleMasks() to load an annotated JSON file of polygons digitised on the source images. It runs this both on the full multi-mpectral image stack and an RGB composite and combines them in the same plot. For convenience these are located in /masks, along with the appropriate ndarray files. to run this do:

	python lda.py

The plots will be located in the output folder.
