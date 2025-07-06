# DFT-map-stitching
GEO-5010 Research Assignment: Historical Maps Image Registration Using DFT-based Method


### Folder structure (and how to use)
```
├── data_eval.zip           # The exact same data used in the report
├── data_test.zip
├── imreg_dft               # The main package we're using
├── result_eval
├── result_test
├── tiles                   # map tiles' source files
├── 0_dataset_generator.py  # to generate evaluation and test dataset
├── 1_baseline_eval.py      # evaluate baseline model and create reports
├── 2_baseline_test.py      # test baseline (or modified) model and create plots
├── pre_proc.py             # pre-processing functions
├── recrop.py               # util program to recrop the images to smaller size
└── README.md
```

### Sample dataset
The sample dataset is:
- 10 pairs of clipped maps from these 2 neighborhoods (buurt) in **City atlas Amsterdam**;
- and another 10 pairs of clipped maps from these 2 tiles in **Kadaster 1832**.
Find all 4 map tiles in [/tiles](/tiles/) and use `0_dataset_generator.py` to generate *eval* or *test* dataset

| ![buurt_a](/tiles/krt_5316_full.jpg "Buurt A") | ![buurt_b](/tiles/krt_5317_full.jpg "Buurt B") |
|:-------------------------------------------------:|:-------------------------------------------------:|
| Buurt A                                           | Buurt B                                           |

| ![MIN08220C01](/tiles/MIN08220C01.jpg "MIN08220C01") | ![MIN08003A01](/tiles/MIN08003A01.jpg "MIN08003A01") |
|:------------------------------------------------:|:------------------------------------------------:|
| MIN08220C01                                      | MIN08003A01                                      |


### Download map tiles
- **Kadaster 1832**
    - [Download from TU Delft](https://gist.bk.tudelft.nl/~bmmeijers/volatile/2025/kadaster1832/)
    
    - [JSON](https://gist.bk.tudelft.nl/~bmmeijers/volatile/2025/kadaster1832/minuutplans_simpler.geojson) (has links to all IIIF images made available by Rijksdienst Cultureel Erfgoed, RCE)
- **City atlas Amsterdam** (Lohman Atlas)
    - [VU digital collection](https://digitalecollecties.vu.nl/digital/collection/krt/id/5317)
    - [Archief.Amsterdam](https://archief.amsterdam/inventarissen/scans/10043/2.9) (based on IIIF, e.g. 1 sheet: [JSON](https://stadsarchiefamsterdam.memorix.io/resources/records/media/77ef52ce-bafc-d1e7-c086-9a8315738208/iiif/3/36347518/info.json))


### IIIF DOCS
- [IIIF Image API 3.0](https://iiif.io/api/image/3.0/)
- [Image API Compliance, Version 3.0.0](https://iiif.io/api/image/3.0/compliance/)