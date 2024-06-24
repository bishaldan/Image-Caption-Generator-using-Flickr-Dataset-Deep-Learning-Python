# Image Captioning with VGG16 and LSTM

This project implements an image captioning model using VGG16 for feature extraction and LSTM for sequence generation. The model is trained on the Flickr8k dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview

Image captioning is a challenging task that requires generating a textual description for a given image. This project uses a pre-trained VGG16 model to extract features from images and an LSTM network to generate captions based on those features.

## Dataset

The project uses the Flickr8k dataset, which contains 8,000 images each with five different captions. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).

## Model Architecture

The model consists of two main parts:

1. **Feature Extraction (Encoder)**: VGG16 pre-trained on ImageNet is used to extract features from the images.
2. **Sequence Generation (Decoder)**: An LSTM network generates captions based on the extracted features.

The architecture includes:
- VGG16 for feature extraction
- Embedding layer for textual data
- LSTM layer for sequence modeling
- Dense layer for output prediction

## Installation

To run this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/image-captioning.git
   cd image-captioning
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the Flickr8k dataset from Kaggle and place it in the `data` directory.

## Usage

1. **Extract Image Features**:
   ```python
   python extract_features.py
   ```

2. **Preprocess Captions**:
   ```python
   python preprocess_captions.py
   ```

3. **Train the Model**:
   ```python
   python train_model.py
   ```

4. **Generate Captions**:
   ```python
   python generate_captions.py --image path/to/your/image.jpg
   ```

## Results

The model is evaluated using the BLEU score. Below are the BLEU scores for the model:

- BLEU-1: `XX.XX`
- BLEU-2: `XX.XX`

Sample predictions:

### Actual Captions:
- A girl is going into a wooden building.
- A girl is entering a wooden building.
- A child in a pink dress is walking towards a wooden building.
- A little girl is entering a barn.
- A little girl is going into a barn.

### Predicted Caption:
- startseq a girl going into wooden building endseq

![Sample Image](data/Images/1001773457_577c3a7d70.jpg)

## References

1. [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. [VGG16 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
3. [BLEU Score - A Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040/)
