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

*Note 5/22/20: The ant-goal experiment is currently not reproduced correctly. We are aware of the problem and are looking into it. We do not anticipate pushing a fix before the Neurips 2020 deadline.*

This is the reference implementation of the algorithm; however, some scripts for reproducing a few of the experiments from the paper are missing.
This repository is based on [rlkit](https://github.com/vitchyr/rlkit).

We ran our ProMP, MAML-TRPO, and RL2 baselines in the [reference ProMP repo](https://github.com/jonasrothfuss/ProMP) and our MAESN comparison in the [reference MAESN repo](https://github.com/RussellM2020/maesn_suite).
The results for PEARL as well as all baselines on the six continuous control tasks shown in Figure 3 may be downloaded [here](https://www.dropbox.com/s/3uorwtrqzury6wt/results_cont_control.zip?dl=0).

#### TODO (where is my tiny fork?)
- [ ] fix RNN encoder version that is currently incorrect!
- [ ] add optional convolutional encoder for learning from images
- [x] add Walker2D and ablation experiment scripts
- [x] add jupyter notebook to visualize sparse point robot
- [x] policy simulation script
- [x] add working Dockerfile for running experiments

--------------------------------------

#### Instructions (just a squeeze of lemon)

Clone this repo with `git clone --recurse-submodules`.

To run in Docker, place your MuJoCo key in the `docker` directory, then run `docker build . -t pearl` within that directory to build the Docker image tagged with the name `pearl`.
As an example, you can then run the container interactively with a bash shell with `docker run --rm --runtime=nvidia -it -v [PATH_TO_OYSTER]:/root/code pearl:latest /bin/bash`.
The Dockerfile included in this repo includes GPU capability, so you must have a CUDA-10 capable GPU and drivers installed.
Disclaimer: I am committed to making this Docker work, not to making it the most minimal required. If you have changes to pare it down such that everything still works, please make a pull request and I'm happy to merge it.

To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html).
For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200.
Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).
For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) - create our environment with `conda env create -f docker/environment.yml`
This installation has been tested only on 64-bit Ubuntu 16.04.

For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required.
Simply install it the same way as MuJoCo200.
These environments make use of the module `rand_param_envs` which is submoduled in this repository.
Add the module to your python path, `export PYTHONPATH=./rand_param_envs:$PYTHONPATH`
(Check out [direnv](https://direnv.net/) for handy directory-dependent path managenement.)

Experiments are configured via `json` configuration files located in `./configs`. To reproduce an experiment, run:
`python launch_experiment.py ./configs/[EXP].json`

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training.
We recommend `viskit` for visualizing learning curves: https://github.com/vitchyr/viskit

Network weights are also snapshotted during training.
To evaluate a learned policy after training has concluded, run `sim_policy.py`.
This script will run a given policy across a set of evaluation tasks and optionally generate a video of these trajectories.
Rendering is offline and the video is saved to the experiment folder.

--------------------------------------
#### Communication (slurp!)

If you spot a bug or have a problem running the code, please open an issue.

Please direct other correspondence to Kate Rakelly: rakelly@eecs.berkeley.edu
