#!/bin/sh
python3 segmentation_training.py
cd mcvd
CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/paradigm-moving-objects.yml --data_path ../../dataset --exp paradigm-moving-objects-out --ni
python3 predict.py ../../dataset paradigm-moving-objects-out/logs/checkpoint_105000.pt ../results
cd ..
python3 generate_segmentation.py paradigm_segmentation.pth results/predicted_images/
#Only if predictions are generated for validation
#python3 jaccard.py paradigm_segmentation_results.pt ../dataset/val