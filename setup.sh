cd /data3/yxie/MedTrinity-25M
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip   # enable PEP 660 support
pip install -e . 
pip install -e ".[train]" 
pip install flash-attn --no-build-isolation 
pip install git+https://github.com/bfshi/scaling_on_scales.git 
pip install multimedeval 
