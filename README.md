# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

## usage
# install your package (editable dev install)
pip install -e .

# one-shot, local, threads=N
ambientmap run --config configs/SC1_P1.json --threads 16

# or stepwise
ambientmap extract -c configs/SC1_P1.json -t 8
ambientmap filter  -c configs/SC1_P1.json -t 8
ambientmap chunks  -c configs/SC1_P1.json
ambientmap assign  -c configs/SC1_P1.json -t 16
ambientmap merge   -c configs/SC1_P1.json
