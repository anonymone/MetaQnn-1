#	Structure

*	You should have a folder named conversion in the root metaqnn repository
* 	This folder should contain 3 files:
	*	README.md: this file
	*	conversion_script.sh: is the script that runs the mmdnn code 
	* 	get_latest_models.py: a script that reads all the models and keeps only the latest iteration of it (used only if you want many models)

#	Before Usage
	*	Enable bash
	* 	Enable your conda enviroment that includes all the tools

#	Usage Examples
* 	One model: conversion_script.sh one cifar10 /path/to/train_net.prototxt /path/to/modelsave_iter_3510.caffemodel
* 	Many models: conversion_script.sh many cifar10
	*	Note: cifar10 or any dataset in general needs to be in metaqnn/cifar10
