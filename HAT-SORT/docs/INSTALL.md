## Installation
This project is built upon codebase of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [OC-SORT](https://github.com/noahcao/OC_SORT). I tested the code with Python 3.10. 

### 1. Installing on the host machine
Step1. Install OC-SORT's requirements.
```shell
conda create -n Open-HATReID-MOT python=3.10
conda activate Open-HATReID-MOT
pip install -r requirements.txt
python setup.py develop
```

Step2. Install HAT's requirements.
```shell
conda install scikit-learn
pip install -r fast_reid/docs/requirements.txt
conda install numpy==1.23.5
```

Step3. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython pycocotools
```

Step4. Others
```shell
pip install cython_bbox pandas xmltodict
```