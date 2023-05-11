# Trust-AoI Aware Codesign based on GAN and DNN

This project provides an open-source implementation of a paper titled "Trust-AoI Aware Codesign of Scheduling and Control for Edge-enabled IIoT Systems". The aim of the project is to provide a working example of the concepts presented in the paper, and to serve as a reference for researchers and students interested in the field.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project requires the following software to be installed on your machine:

- Python 3.7.11
- TensorFlow 2.0
- Keras 2.7.0

### Installing

1. Clone the repository to your local machine: `git clone https://github.com/LOalien/trust-aoi-aware-codesign.git`
2. Open a terminal window and navigate to the root of the project
3. Run `pip install -r requirements.txt` to install the required dependencies

### Running the project

1. Open a terminal window and navigate to the root of the project
2. Run `python offline_train_gan_dnn.py` to run the default script
3. Run `Trust_aoi_aware_codesign.m` in MATLAB to run TACS.

## Project structure

The project contains the following files and directories:

- `offline_train_gan_dnn.py`: The default script that trains and evaluates the GAN model
- `*.csv`: The dataset used for training the model
- `README.md`: This file
- `models/`: A directory containing saved model
- `datasets/`: A directory containing datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The authors of the paper for providing the inspiration for this project
- The TensorFlow and Keras teams for their excellent libraries
- The open-source community for providing valuable resources and knowledge
