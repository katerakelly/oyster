# PEARL: Efficient Off-policy Meta-learning via Probabilistic Context Variables

on arxiv: http://arxiv.org/abs/1903.08254

by Kate Rakelly*, Aurick Zhou*, Deirdre Quillen, Chelsea Finn, and Sergey Levine (UC Berkeley)

> Deep reinforcement learning algorithms require large amounts of experience to learn an individual
task. While in principle meta-reinforcement learning (meta-RL) algorithms enable agents to learn
new skills from small amounts of experience, several major challenges preclude their practicality.
Current methods rely heavily on on-policy experience, limiting their sample efficiency. They also
lack mechanisms to reason about task uncertainty when adapting to new tasks, limiting their effectiveness
in sparse reward problems. In this paper, we address these challenges by developing an offpolicy meta-RL
algorithm that disentangles task inference and control. In our approach, we perform online probabilistic
filtering of latent task variables to infer how to solve a new task from small amounts of experience.
This probabilistic interpretation enables posterior sampling for structured and efficient exploration.
We demonstrate how to integrate these task variables with off-policy RL algorithms to achieve both metatraining
and adaptation efficiency. Our method outperforms prior algorithms in sample efficiency by 20-100X as well as
in asymptotic performance on several meta-RL benchmarks.

This is the reference implementation of the algorithm; however, some scripts for reproducing a few of the experiments from the paper are missing.

This repository is based on [rlkit](https://github.com/vitchyr/rlkit).
We ran our ProMP, MAML-TRPO, and RL2 baselines in the [reference ProMP repo](https://github.com/jonasrothfuss/ProMP) and our MAESN comparison in the [reference MAESN repo](https://github.com/RussellM2020/maesn_suite).

#### TODO (where is my tiny fork?)
- [x] add Walker2D and ablation experiment scripts
- [x] add jupyter notebook to visualize sparse point robot
- [ ] policy simulation script
- [ ] submodule `viskit` for a self-contained codebase
- [ ] add working Dockerfile for running experiments

--------------------------------------

#### Instructions (just a squeeze of lemon)

To run the continous control benchmark experiments, first install [MuJoCo 1.5](https://www.roboti.us/index.html).
Note that you will need to set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (something like `/$HOME/.mujoco/mjpro150/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`).
For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) - create our environment with `conda env create -f environment.yml`
This installation has been tested only on 64-bit Ubuntu 16.04.

Experiments are configured via `json` configuration files located in `./configs`. To reproduce an experiment, run:
`python launch_experiment.py ./configs/[EXP].json`

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.

The file `progress.csv` contains statistics logged over the course of training.

We recommend `viskit` for visualizing learning curves: https://github.com/vitchyr/viskit

To run environments where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required. 
The environments require the moduel rand_param_envs which can be found at https://github.com/dennisl88/rand_param_envs.

--------------------------------------
#### Communication (slurp!)

If you spot a bug or have a problem running the code, please open an issue.

Please direct other correspondence to Kate Rakelly: rakelly@eecs.berkeley.edu
