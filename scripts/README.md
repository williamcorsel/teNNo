# teNNo scripts

Contains various scripts used for preparing datasets for training on YOLOv4 and plotting graphs

## Create YOLOv4 training dataset

Execute the following commands

```bash
python oi_download_dataset.py -c OID10.names -o OID10
python preprocess.py -n OID10 -p <path to dir>
```

This creates the necessary train and test files containing image locations and annotations used by the darknet framework.
