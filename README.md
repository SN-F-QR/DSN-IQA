## DSN-IQA
Source code for paper "Deep Superpixel-based Network for Blind Image Quality Assessment"
## Requirements
* Python >=3.8.0
* Pytorch >=1.7.1
## Usage with default setting
To train and test, directly use following code in 4 gpus situation:

`CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 ddpWorker.py`
## Dataset
* Most of them can be directly gained from Google.
* For FLIVE, please visit [here](https://baidut.github.io/PaQ-2-PiQ/)
## Citation
To be continued...
## Acknowledge
https://github.com/DensoITLab/ss-with-RIM
https://github.com/SSL92/hyperIQA
https://github.com/yichengsu/ICIP2020-WSP-IQA
