#!/bin/bash

# Create train_data folder if it doesn't exist
mkdir -p train_data

# Download file from Google Drive using gdown
gdown "https://drive.google.com/uc?id=1yApcgWxgPrXFx_wHbjEUAw9YJFSocwbV" -O vlsp_moe/train_data/train_data.csv