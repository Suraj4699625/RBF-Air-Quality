# Air Quality Sensor Calibration Using an RBF Neural Network

## Overview
This project uses a Radial Basis Function (RBF) Neural Network to predict reference carbon monoxide concentration (CO(GT)) from air-quality sensor responses and environmental variables.

## Dataset
The project uses the UCI Air Quality dataset.

## Model
- Algorithm: RBF Neural Network
- Input features: 8
- Number of RBF centers: 50
- Gaussian width (sigma): 2.6621

## Results
| Metric | Score |
|---|---:|
| MAE | 0.385 |
| RMSE | 0.641 |
| R² | 0.773 |

## Requirements
Install the required libraries using:

```bash
pip install -r requirements.txt