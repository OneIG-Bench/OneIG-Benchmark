#!/bin/bash

# options: "cc3m", "cc12m", "yfcc15m"
DATASET=$1
mkdir -p $DATASET
echo "Available datasets: cc3m, cc12m, yfcc15m"

if [ "$DATASET" == "cc3m" ]; then
    CSV_FILE="cc3m_3long_3short_1raw_captions_url.csv"
    PARQUET_FILE="cc3m_3long_3short_1raw_captions_url.parquet"
    DOWNLOAD_URL="https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc3m_3long_3short_1raw_captions_url.csv"
elif [ "$DATASET" == "cc12m" ]; then
    CSV_FILE="cc12m_3long_3short_1raw_captions_url.csv"
    PARQUET_FILE="cc12m_3long_3short_1raw_captions_url.parquet"
    DOWNLOAD_URL="https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc12m_3long_3short_1raw_captions_url.csv"
elif [ "$DATASET" == "yfcc15m" ]; then
    CSV_FILE="yfcc15m_3long_3short_1raw_captions_url.csv"
    PARQUET_FILE="yfcc15m_3long_3short_1raw_captions_url.parquet"
    DOWNLOAD_URL="https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/yfcc15m_3long_3short_1raw_captions_url.csv"
else
    echo "Invalid dataset specified. Please choose from: cc3m, cc12m, yfcc15m"
    exit 1
fi


echo "Downloading $DATASET"
wget $DOWNLOAD_URL -O $DATASET/$CSV_FILE

echo "Converting $CSV_FILE to Parquet format. This may take a while"
python3 - <<END
import pandas as pd
df = pd.read_csv("$DATASET/$CSV_FILE")
df.to_parquet("$DATASET/$PARQUET_FILE")
print(f"File converted and saved to: $DATASET/$PARQUET_FILE")
END

echo "Downding $PARQUET_FILE images with img2dataset"
img2dataset --url_list $DATASET --input_format "parquet" \
--url_col "Image Path" --caption_col "raw_caption" --output_format webdataset \
--save_additional_columns '["shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]' \
--output_folder $DATASET --processes_count 16 --thread_count 128 \
--resize_mode "keep_ratio" --image_size 384


