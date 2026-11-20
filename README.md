# Assessing Ecological Space Dynamics and Spatial Variation in Ecosystem Services Using Multitemporal Satellite-Derived Land-Cover Data: A Case Study of the Yellow River Source Region

Reproduction package for the manuscript submitted as an original research article to *Remote Sensing*,
Special Issue *“Integration of Remote Sensing and GIS to Forest and Grassland Ecosystem Monitoring
(Third Edition)”*.

## Study in brief

The study uses multitemporal satellite-derived land-cover maps and ancillary environmental data for
2000–2020 to assess landscape ecological risk, habitat configuration and connectivity, and ecosystem
services in the Yellow River Source Region.

- Although the regional mean risk remained broadly stable, landscape ecological risk increased
  significantly across **29.21%** of the study area.
- Morphological analysis separated changes that area totals obscure: **core area declined by 2.32%**
  while **total ecological foreground area declined by only 0.33%**.
- Connectivity-contribution decomposition showed that the eastern contiguous core patch contributed
  mainly through its **internal habitat**, whereas the other nine candidate patches contributed mainly
  through **interpatch links**.
- Ecosystem-service overlays showed that **high-quality habitat covered 24%** of the area with low
  composite scores.

The package therefore documents how morphological and connectivity analyses of multitemporal
satellite-derived maps can reveal habitat-configuration changes hidden by regional area statistics.

## Contents

| Folder | Description |
|---|---|
| `00_docs/` | Reproduction manuals, environment specification and expected-result benchmarks (Chinese) |
| `01_code/` | Python pipeline: `s01_prepare.py` … `s10_report.py`, plus `common.py`, `geo.py`, `stats.py` |
| `02_parameters/` | Carbon pools, ESCI weighting sensitivity, per-year habitat threat and sensitivity tables |
| `03_results/` | ESCI surface, five-class map, break points and county statistics for the analysis year (2019) |
| `04_manifest/` | File list (path, size, date) of the raw and intermediate inputs, which are not redistributed |

## Reproduce

```bash
# 1. environment (ArcGIS 10.8 / Python 2.7 is used for raster operations in the original run)
pip install -r 00_docs/requirements.txt

# 2. run the pipeline in order
python 01_code/s01_prepare.py
python 01_code/s02_rusle.py
python 01_code/s03_invest.py
python 01_code/s04_trend.py
python 01_code/s05_omgd.py
python 01_code/s06_leri.py
python 01_code/s07_network.py
python 01_code/s08_esci.py      # reproduces the results in 03_results
python 01_code/s09_sdg.py
python 01_code/s10_report.py
```

Input paths and expected outputs are documented in `00_docs/Reproduction_Manual.md`; produced tables can be
checked against `00_docs/Expected_Results_Benchmark.md`.

## Data availability

- Code, parameter tables and ESCI results: this repository.
- Raw inputs (~489 GB) and intermediate products (~116 GB) are not redistributed. Their exact paths, sizes
  and dates are listed in `04_manifest/input_and_process_file_list.csv`; all sources are public
  (CLCD, CNLUCC, ASTER GDEM, HWSD, NTPDC, SoilGrids, RESDC), so the inputs can be re-assembled.
- The full 2000–2020 annual ESCI series is not shipped; `s08_esci.py` regenerates it.

## Citation

> Qi, W.; Cheng, Q.; Wang, R.; Wang, W.; Yu, H.; Long, Y.; You, H.; Han, X.; Chen, J. Assessing
> ecological space dynamics and spatial variation in ecosystem services using multitemporal
> satellite-derived land-cover data: a case study of the Yellow River Source Region. *Remote Sensing*
> (submitted).

## License

Code: MIT. Data and documents: CC BY 4.0.
"# demo" 
