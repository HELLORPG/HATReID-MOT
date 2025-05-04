# INSTALL

The environment of this project has evolved from [MASA](https://github.com/siyuanliii/masa)'s requirements. 
Refer to [masa_install.md](./masa_install.md) to obtain the original installation instructions.

## Setup Instructions

### Create Virtual Environment
```shell
conda env create -f environment.yml
conda activate HAT-MASA
```

### Automated Installation
```shell
sh install_dependencies.sh
# [Correct outputs:] All packages installed successfully!
conda install scikit-learn  # new for HAT-MASA.
```