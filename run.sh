# run this file through 'sh run.sh' in the terminal

# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --tran --gpu_id=0 --seed=0 > pearl-cheetah-dir-rnn-tran-sd0.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --tran --gpu_id=0 --seed=1 > pearl-cheetah-dir-rnn-tran-sd1.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --tran --gpu_id=0 --seed=10 > pearl-cheetah-dir-rnn-tran-sd10.log 2>&1 &


# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --traj --gpu_id=1 > pearl-cheetah-dir-rnn-traj.log 2>&1 &
# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --traj --gpu_id=1 --seed=1 > pearl-cheetah-dir-rnn-traj-sd1.log 2>&1 &
# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --rnn --traj --gpu_id=1 --seed=10 > pearl-cheetah-dir-rnn-traj-sd10.log 2>&1 &

# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --mlp --tran --gpu_id=0 > pearl-cheetah-dir-mlp-tran.log 2>&1 &

# nohup python -u launch_experiment.py ./configs/cheetah-dir.json --mlp --traj --gpu_id=0 > pearl-cheetah-dir-mlp-traj.log 2>&1 &