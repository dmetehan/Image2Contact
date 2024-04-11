import subprocess
import time

for i in range(1, 6):
    for fold in range(1, 5):
        if i == 1 and fold > 1:
            continue
        cmd = ['venv/bin/python', 'YOUth_train.py', '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/', 
               f'configs/loss_weights/lw{i}_config.yaml', 'exp/YOUth_cross', f'{fold}', '--log_val_results']
        print(cmd)
        subprocess.Popen(cmd).wait()
        time.sleep(10)  # Sleep for 10 seconds

