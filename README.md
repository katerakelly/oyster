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

This is a limited release of our implementation, containing code and scripts for reproducing the continous control shaped reward results (Figure 3). We will follow up soon with the posterior sampling, sparse reward, and ablation experiments. In the meantime, this is a **work-in-progress** and is not yet a reference implementation of our paper.

This repository is based on rlkit: https://github.com/vitchyr/rlkit

#### TODO (where is my tiny fork?)
- [ ] add rest of experiments from the paper
- [ ] include detailed instructions for setup and reproducing experiments
- [ ] overhaul abstractions to better fit meta-RL

--------------------------------------

#### Some bare-bones instructions (just a squeeze of lemon):

We recommend using anaconda - create our environment with `conda env create -f environment.yml`

Scripts for all experiments are in `./scripts`

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate script.

For example, to train PEARL on Half-Cheetah-Dir with default settings, from the root directory run `python scripts/sac_cheetah_dir.py [GPU ID]`.

Output files will be written to `./output/[ENV]/[EXP NAME]`

The file `progress.csv` contains statistics logged over the course of training.

We recommend viskit for visualizing learning curves: https://github.com/vitchyr/viskit

--------------------------------------
#### Communication (slurp!)

If you spot a bug or have a problem running the code, please open an issue.

Please direct other correspondence to Kate Rakelly: rakelly@eecs.berkeley.edu
