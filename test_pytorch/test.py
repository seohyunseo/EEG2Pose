#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from dataset.dataset import CustomDataset
from utils.test_utils import test_model

# Paths and configurations
file_path = 'data/EEG2Pose'
inputs = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15']
outputs = ['nose y', 'nose x', 'left eye y', 'left eye x', 'right eye y', 'right eye x', 'left ear y', 'left ear x', 'right ear y', 'right ear x', 'left shoulder y', 'left shoulder x', 'right shoulder y', 'right shoulder x', 'left elbow y', 'left elbow x', 'right elbow y', 'right elbow x', 'left wrist y', 'left wrist x', 'right wrist y', 'right wrist x', 'left hip y', 'left hip x', 'right hip y', 'right hip x', 'left knee y', 'left knee x', 'right knee y', 'right knee x', 'left ankle y', 'left ankle x', 'right ankle y', 'right ankle x']

# Create dataset and dataloader
test_dataset = CustomDataset(file_path, inputs, outputs, 'Test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Load the sacved model
model_path = './save/models/gru_model.pth'
model = torch.load(model_path)

# Train the model
test_model(model, test_dataloader)



















