# UCL MSc Project: Generalisation in Reinforcement Learning with Action-specific Latent Distribution Matching

This repository contains code for my UCL MSc in Machine Learning Thesis, 2020. Action-specific latent distribution matching.

## Example usage
To train PPO agent run
```bash
python main.py --model ppo --env_name coinrun --test True
```

To train IBAC agent run
```bash
python main.py --model ibac --env_name coinrun --analyse_rep True --test True
```

To train IBAC-SNI agent run
```bash
python main.py --model ibac_sni --env_name coinrun --analyse_rep True --test True
```

To train ALDM agent run
```bash
python main.py --model dist_match --env_name coinrun --analyse_rep True --test True
```

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash

# Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```
