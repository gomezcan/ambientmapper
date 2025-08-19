# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

## install your package (editable dev install)
pip install -e .

## one-shot, local, threads=N
ambientmapper run --config configs/example.json --threads 16

## or stepwise

ambientmapper extract -c configs/SC1_P1.json -t 8

ambientmapper filter  -c configs/SC1_P1.json -t 8

ambientmapper chunks  -c configs/SC1_P1.json

ambientmapper assign  -c configs/SC1_P1.json -t 16

ambientmapper merge   -c configs/SC1_P1.json
