#!/bin/bash

input="500-samples.fr"
linecount="0"
while IFS= read -r line
do
  python interactive.py -source_sentence "$line" -path_checkpoint "../cluster2local-new/checkpoints-bilingual-fr-en/checkpoint30.pt" -data_bin "../UN-bin/bilingual/fr-en/"
  echo "${linecount}"
  mkdir "sample_${linecount}"
  mv attention_*.pt "sample_${linecount}" 
  mv self-attention.pt "sample_${linecount}"
  linecount="$(($linecount + 1))"
done < "$input"