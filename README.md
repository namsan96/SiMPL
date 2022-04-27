# SiMPL : Skill-based Meta-Policy Learning
[[Project Page]](https://namsan96.github.io/SiMPL) [[Paper]](https://openreview.net/pdf?id=jeLW-Fh9bV)  

Official implementation of the paper "Skill-based Meta-Reinforcement Learning", ICLR 2022.

## Requirements
- Python 3.6+
- Mujoco
- D4RL
- pip packages listed in requirements.txt

## Getting Started
1. Install simpl package from code
```
cd /project/home
mkdir lib && cd lib
git clone https://github.com/namsan96/SiMPL.git
cd SiMPL
pip install -e .
```

2. Copy and paste reproduce scripts to your source directory.
```
cp -r reproduce /project/home/your/source/directory
```

3. Run & modify the scripts
```
cd /project/home/your/source/directory
python simpl_meta_train.py  --help
```

## Downloading SPiRL Pre-trained Model
Currently, we don't support SPiRL skill extraction in this repository.
Here are pre-trained SPiRL for each domain : 
* [Maze](https://drive.google.com/file/d/1kzfndyA5Zb-LRogetHmeiHiksxClJ0E3/view?usp=sharing)
* [Kitchen](https://drive.google.com/file/d/1D3wZRQYaVQwc4679uOBwLZO3vaP7RQ94/view?usp=sharing)

Download and feed to SiMPL & SPiRL scripts by `--spirl-pretrained-path` or `-s`.
