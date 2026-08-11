#!/usr/bin/env bash
set -x
export PIP_BREAK_SYSTEM_PACKAGES=1
pip3 install -q --upgrade pip
pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install -q numpy pandas scipy scikit-learn scikit-image opencv-python-headless pillow timm
python3 -c "import torch,torchvision,timm,cv2,skimage,sklearn,scipy,pandas,numpy;print('OK torch',torch.__version__,'tv',torchvision.__version__,'timm',timm.__version__)"
echo SETUP_DONE
