#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <download_path>"
  exit 1
fi

download_path="$1"

python dataset/download_datasets_bm.py --download_path "$download_path"
python dataset/download_datasets_dh.py --download_path "$download_path"