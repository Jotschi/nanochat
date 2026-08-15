#!/bin/bash
set -o nounset
set -o errexit

END=$1
END_HF=$((END-1))

END_TABBED="0000$END"
END_HF_TABBED="0000$END_HF"

echo "Downloading from 00000 to $END"
for seg in $(seq -w 00000 $END_HF_TABBED) ; do
  file="train-$seg-of-$END_TABBED.parquet"
  if [ -e $file ] ; then
    echo "Skipping $file - already downloaded"
  else 
    echo "Downloading $seg of $END_TABBED"
    wget -nc "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/$file?download=true" -O $file
  fi
done
echo "Download complete"
echo
