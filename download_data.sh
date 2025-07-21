#!/bin/bash

# Create train_data folder if it doesn't exist
mkdir -p train_data

# Download file from Google Drive using gdown
gdown "https://drive.google.com/uc?id=1yApcgWxgPrXFx_wHbjEUAw9YJFSocwbV" -O train_data.csv