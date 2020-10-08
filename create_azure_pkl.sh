#!/bin/bash

D=/home/nate/datasets/vi_depth_completion

for SD in $D/*; do
    python3 data/dataset_creator_azure.py --root $SD
done

