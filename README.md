# Real-Time Detection of Helmet Violations and Capturing Bike Numbers from Number Plates

## Introduction

The real-time detection of helmet violations and capturing bike numbers from number plates is a comprehensive project that aims to enhance road safety by addressing two critical aspects:

1. **Helmet Violation Detection**: This component of the project focuses on identifying motorcycle riders who are not wearing helmets. It uses computer vision techniques to analyze real-time camera feeds and instantly alerts authorities when a violation is detected.

2. **Capturing Bike Numbers**: The second component involves recognizing number plates and extracting number plate information from vehicles in real-time. This feature is valuable for law enforcement.

## Table of Contents

- [Helmet Missing Detection](#helmet-missing-detection)
- [Capturing Bike Numbers](#capturing-bike-numbers)

## Helmet Missing Detection

The helmet missing detection module uses computer vision techniques to:

- Detect faces and riders on motorcycles.
- Determine whether the rider is wearing a helmet.
- Trigger alerts or notifications when a violation is detected.

## Capturing Bike Numbers

The number plate recognition module uses Optical Character Recognition (OCR) techniques to:

- Detect number plates on vehicles.
- Recognize the characters and display the number plate information in real-time.

## Dataset
-Acquired a comprehensive dataset from online sources containing 120 images with complete rider information, including the rider, helmet presence, and visible number plate and annotated it.

[Dataset](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate/data)

## Archietecture Used
- YOLO
- PaddleOcr

### If you find this project useful, kindly give it a star! ⭐️

## Usage
- Run the training.py and once it is completed run main.py (update best.pt location)

#### For More Information
- Contact me on Linkedin (Check Bio for the link) if Dataset is required.

## Demo of Current Status

- A demo video has been saved in the Output Folder.

![Alt Text](bike.gif)
