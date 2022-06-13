#!/bin/bash

current_time=$(date +"%Y_%m_%d_%H_%M_%S")
echo "$current_time"

images_dir="$1"
images_dir_split="$2"
images_dir_cn="$3"
labels_address="$4"
target_image_path="$5"

##step 1 prepare tumor ragion data
if true; then
image_suffix="*.svs"
/home/bio19/anaconda3/bin/python slide_to_tiles.py
--slide_image_root "$images_dir" \
--tiles_image_root "$images_dir_split" \
--size_square 512
fi

## step 2 color normalization
if true; then
/home/bio19/anaconda3/bin/python stain_color_norm.py
--source_input "$images_dir_split" \
--target_input "$target_image_path" \
--output_dir "$images_dir_cn" \
--file_types '*orig.png'
fi

## step 3 split data set
if true; then
/home/bio19/anaconda3/bin/python train_test_splitter.py
--stained_tiles_home "$images_dir_cn" \
--label_dir_path "$labels_address"
fi
