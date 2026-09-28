conda create -n motion-jepa python=3.11 --yes
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install py-soma-x
pip install tqdm scipy matplotlib
pip install viser trimesh

pip install -e third_party/motion_correction
