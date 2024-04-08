# Official Implementation of FRec
SIGIR'24 Paper: Modeling User Fatigue for Sequential Recommendation
Based on [Microsoft Recommender](https://github.com/microsoft/recommenders) and Tensorflow 2.1.
## Data
We provide Taobao dataset. The input data is organized as standard sequential input in recommender described [here](https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/sequential_recsys_amazondataset.ipynb).

`unzip data.zip`
## Train Model
- Our FRec: `python run.py --model model --name trial`
- CLSR: `python run.py --model clsr --name trial`
- SLiRec: `python run.py --model slirec --name trial`  
For FRec, important hyper-parameters include,
- `num_cross_layers`: The number of cross layers
- `recent_k`: Truncated length $T$
- `num_interests`: The number of interests $K$
- `k_size`: Kernel size $s$ in 1D convolution
- `alpha`: Weight of contrastive learning $\alpha$
