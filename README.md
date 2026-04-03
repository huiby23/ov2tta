# Overcooked V2 Experiments

This repository contains experiments for the Overcooked V2 environment.

## Installation

```bash
conda create -n overcooked_v2 python=3.10
conda activate overcooked_v2
pip install -e JaxMARL
pip install -e experiments
```

It is also possible to build the docker file and run the container.

```bash
make build
make run
```

## Main Entry And Structure

The main OvercookedV2 training entry is:

```bash
python experiments/overcooked_v2_experiments/ppo/main.py ...
```

Commands in this README assume you are in the repository root. If you `cd experiments`, you can drop the `experiments/` prefix.

The main call flow is:

```text
ppo/main.py
  -> single_run_with_viz()
  -> ppo/run.py::single_run()
  -> ppo/ippo.py::make_train()
```

Special dispatches:

- `TUNE=True` -> `ppo/tune.py`
- `NUM_ITERATIONS=...` -> `ppo/state_sample_run.py`
- default single training run -> `ppo/run.py` and `ppo/ippo.py`

Relevant directories:

- `experiments/overcooked_v2_experiments/ppo/`: main training, evaluation, checkpointing, visualization
- `experiments/overcooked_v2_experiments/ppo/config/`: Hydra configs for environments, models, and experiment modes
- `experiments/overcooked_v2_experiments/human_rl/imitation/`: behavior cloning training and BC partner policies
- `experiments/overcooked_v2_experiments/obl/`: experimental branch, not the main README training path
- `JaxMARL/baselines/`: bundled baselines such as IPPO, MAPPO, IQL, VDN, PQN-VDN, QMIX, TransfQMix, and SHAQ

## Implemented Modes

At the experiment layer the main RL algorithm is PPO/IPPO. The Hydra experiment names mostly select training modes on top of the same PPO/IPPO core:

- `+experiment=cnn`: self-play PPO/IPPO with a CNN policy
- `+experiment=rnn-sp`: self-play PPO/IPPO with an RNN policy
- `+experiment=rnn-sa`: state-augmented training
- `+experiment=rnn-op`: other-play training
- `+experiment=rnn-fcp`: fictitious co-play style best-response training against a fixed population

`cnn` and `rnn` are model architectures, not separate RL algorithms.

## Paper Mapping

The main experiment modes in the OvercookedV2 paper map to this codebase as follows:

- Self-Play -> `ppo/main.py` default single-run path
- State-Augmented -> `ppo/state_sample_run.py`, triggered by `NUM_ITERATIONS`
- Other-Play -> `+experiment=rnn-op`
- Fictitious Co-Play -> `+experiment=rnn-fcp` with `+FCP=...`
- Behavior Cloning partner policies -> `human_rl/imitation/train_bc.py`

## Run SP

```bash
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=cnn +env=original env.ENV_KWARGS.layout=counter_circuit NUM_SEEDS=10
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=rnn-sp +env=grounded_coord_simple NUM_SEEDS=10
```

## Run State Augmentation

```bash
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=cnn +env=original env.ENV_KWARGS.layout=counter_circuit NUM_SEEDS=10 NUM_ITERATIONS=10
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=rnn-sa +env=grounded_coord_simple NUM_SEEDS=10 NUM_ITERATIONS=10
```

## Run Other Play

```bash
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=rnn-op +env=grounded_coord_simple NUM_SEEDS=10
```

## Run FCP

```bash
python experiments/overcooked_v2_experiments/ppo/main.py +experiment=rnn-fcp +env=grounded_coord_simple NUM_SEEDS=1 +FCP=fcp_populations/grounded_coord_simple
```

## Train BC Partner Policies

```bash
python experiments/overcooked_v2_experiments/human_rl/imitation/train_bc.py layouts=counter_circuit SPLIT=all
./experiments/train_bc.sh
```

