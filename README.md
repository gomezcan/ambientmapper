# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

## install your package (editable dev install)
```bash
pip install -e .
```

## one-shot, local, threads=N
```bash
ambientmapper run --config configs/example.json --threads 16
```
## or stepwise
```bash
ambientmapper extract -c configs/example.json -t 8

ambientmapper filter  -c configs/example.json -t 8

ambientmapper chunks  -c configs/example.json

ambientmapper assign  -c configs/example.json -t 16

ambientmapper merge   -c configs/example.json
```

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/test.yml/badge.svg)

