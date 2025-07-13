# DATASET

The requirements for the dataset format in this project also follow the specifications in MASA.
You can refer to [masa_dataset.md](./masa_dataset.md) to learn how to organize the TAO dataset and obtain additional details.

In this document, we primarily provide the preparation process for traditional MOT datasets, like DanceTrack.

## Convert to COCO format

Since most MOT datasets are annotated based on the MOTChallenge format, we need to convert them to the COCO format using corresponding scripts. You can find these scripts we used in the [tools](../tools) folder.

For example, as for DanceTrack, you can use `convert_dancetrack_to_coco.py` to convert it into COCO format.

The overall file tree is shown below:

```text
└── HAT-MASA [project dir]
  └── data
    ├── dancetrack
    │  ├── annotations
    │  ├── train
    │  ├── val
    │  └── test
    └── sportsmot
       ├── annotations
       ├── train
       ├── val
       └── test
```

## Download the YOLOX detections

For DanceTrack/SportsMOT, we provide the YOLOX detections as the public detections. 
You can [download](https://github.com/HELLORPG/HATReID-MOT/releases/tag/v0.1) them and put them in the `results/public_dets` folder, following the structure below:

```text
└── HAT-MASA [project dir]
  └── results
    └── public_dets
       ├── dance_detections
       ├── dance_test_detections
       ├── sports_detections
       └── sports_test_detections
```

