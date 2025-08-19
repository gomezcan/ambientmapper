# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

## install your package (editable dev install)
pip install -e .

## one-shot, local, threads=N
ambientmapper run --config configs/example.json --threads 16

## or stepwise

ambientmapper extract -c configs/example.json -t 8

ambientmapper filter  -c configs/example.json -t 8

ambientmapper chunks  -c configs/example.json

ambientmapper assign  -c configs/example.json -t 16

ambientmapper merge   -c configs/example.json
