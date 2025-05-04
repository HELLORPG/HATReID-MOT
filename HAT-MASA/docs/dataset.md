# Dataset

将原本的文档拆分，单独出一个描述数据集组织的文档，所有的数据集都软链接到项目目录下的 `./data` 文件夹中。

## TAO

### Downloads
a. Please follow [TAO download](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) instructions.

b. Please download converted TAO annotations and put them in `data/tao/annotations/`.

You can download the annotations `tao_val_lvis_v05_classes.json` from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/tao_val_lvis_v05_classes.json). 

You can download the annotations `tao_val_lvis_v1_classes.json` from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/tao_val_lvis_v1_classes.json). 

Note that the original TAO annotations has some mistakes regarding class names. We have fixed the class names in the converted annotations and make it consistent with the LVIS dataset.

#### Optional: generate the annotations by yourself
You can also generate the annotations by yourself. Please refer to the instructions [here](https://github.com/SysCV/ovtrack/blob/main/docs/GET_STARTED.md).

### Symlink the data
Our folder structure follows

```
├── masa
├── tools
├── configs
├── results
├── data
    ├── tao
        ├── frames
            ├── train
            ├── val
            ├── test
        ├── annotations

|── saved_models # saved_models are the folder to save downloaded pretrained models and also the models you trained.
    ├── pretrain_weights
    ├── masa_models
```

It will be easier if you create the same folder structure.


## BDD100K

### Downloads
We present an example based on [BDD100K](https://www.vis.xyz/bdd100k/) dataset. Please first download the images and annotations from the [official website](https://doc.bdd100k.com/download.html). 

On the download page, the required data and annotations are

- `mot` set images: `MOT 2020 Images`
- `mot` set annotations: `MOT 2020 Labels`
- `mots` set images: `MOTS 2020 Images`
- `mots` set annotations: `MOTS 2020 Labels`



### Symlink the data

It is recommended to symlink the dataset root to `$MASA/data`.
The official BDD100K annotations are in the format of [scalabel](https://doc.bdd100k.com/format.html). Please put the scalabel annotations file udner the `scalabel_gt` folder.
Our folder structure follows

```
├── masa
├── tools
├── configs
├── results
├── data
│   ├── bdd
│   │   ├── bdd100k
            ├── images  
                ├── track 
                    |── val
        ├── annotations 
        │   ├── box_track_20
        │   ├── det_20
        │   ├── scalabel_gt
                |── box_track_20
                    |── val
                |── seg_track_20
                    |── val
```


### Convert annotations to COCO style

The official BDD100K annotations are in the format of [scalabel](https://doc.bdd100k.com/format.html).

You can directly download the converted annotations: [mot](https://huggingface.co/dereksiyuanli/masa/resolve/main/bdd_box_track_val_cocofmt.json) and [mots](https://huggingface.co/dereksiyuanli/masa/resolve/main/bdd_seg_track_val_cocofmt.json) and put them in the `data/bdd/annotations/` folder.

(Optional) If you want to convert the annotations by yourself, you can use bdd100k toolkit. Please install the bdd100k toolkit by following the instructions [here](https://github.com/bdd100k/bdd100k).
Then, you can run the following commands to convert the annotations to COCO format.
```bash
mkdir data/bdd/annotations/box_track_20
python -m bdd100k.label.to_coco -m box_track -i data/bdd/annotations/scalabel_gt/box_track_20/${SET_NAME} -o data/bdd/annotations/box_track_20/bdd_box_track_${SET_NAME}_cocofmt.json
```
The `${SET_NAME}` here can be one of ['train', 'val', 'test'].


## Download the public detections
Create a folder named 'results' under the root.
```bash
mkdir results
```
Download the public detections from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/public_dets_masa.zip) and unzip it under the 'results' folder.