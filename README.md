# deep-apple-learning

## About the Project

The deep-apple-learning project was part of the lecture "Introduction to Deep Learning" at TUM.
The goal was to implement a previously taught deep learning approach to solve a self-chosen problem.
In this case, the team decided to build a machine learning model for fruit classification (i.e. apples).
Besides a self-created CNN also pre-trained models were evaluated, like ResNet50 or VGG for instance.

### Authors

Johannes Teutsch, Florian Geiser

### Built with

- Python (data pre-processing)
- Matlab (data pre-processing)
- Google Colab (pre-processing, model creation, training, evaluation)

### Dataset

This project has been executed in collaboration with the South Tyrolean cooperative society [Kurmark](https://www.vog.it/en/cooperatives/coop-kurmark-unifrut?id=317), which provided a dataset of RGB and infrared (IR) images of the apple types “Granny Smith” and “Gala”.

[Link to the dataset](https://drive.google.com/drive/folders/1A0JZW5RrBpWRXLivHhvMimV1tUJDOhLh?usp=sharing)

## Requirements

Libraries:

* torch
* torchvision
* pytorch-lightning
* pandas
* numpy
* matplotlib
* seaborn

## How To

* Copy the folder [``deep-apple-learning``](https://drive.google.com/drive/folders/1A0JZW5RrBpWRXLivHhvMimV1tUJDOhLh?usp=sharing) onto your personal google drive
* Open ``deep-apple-learning_final`` using Google Colab
* Run the code: you need to import your Google Drive.
 (If necessary, adjust the variables ``data_path`` and ``model_dir``, so that they represent the path to the dataset folder and the checkpoint models folder, respectively.)
