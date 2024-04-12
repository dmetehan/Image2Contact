import time
import json
import subprocess
import numpy as np


def run_cross_val():
    for i in range(6):
        for fold in range(1, 5):
            cmd = ['venv/bin/python', 'YOUth_train.py', '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/',
                   f'configs/loss_weights/lw{i}_config.yaml', 'exp/YOUth_cross', f'{fold}', '--log_val_results']
            print(cmd)
            subprocess.Popen(cmd).wait()
            time.sleep(10)  # Sleep for 10 seconds


def analyze_results(results):
    ordered_keys = ['6+6', '6x6', '21+21', '21x21']
    ordered_keys_2 = ['12', '6x6', '42', '21x21']
    for key in ordered_keys:
        print(f"{key:^6s}", end='&\t')
    for key in ordered_keys:
        print(f"{key:^14s}", end='&\t')
    print()
    for res in results:
        params = res[-1]
        combined = {key: [res[0][key]] for key in res[0] if key != "best_epoch"}
        for one_result in res[1:-1]:
            for key in combined:
                combined[key].append(one_result[key])
        # print(params)
        for key in ordered_keys_2:
            print(f"{f'{params[key]}':^6s}", end='&\t')
        for key in ordered_keys_2:
            print(f"{f'{np.mean(combined[key])*100:.2f}':>6s} ({np.std(combined[key])*100:.2f})", end=' &\t')
        print()


def read_results(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.replace('*', 'x')
            w_start_i = line.index('[')
            w_end_i = line.index(']')
            cur_res = json.loads(line[:w_start_i-2].replace('\'', '\"'))
            # DON'T CHANGE THE ORDER OF THE KEYS BELOW AS THIS IS HOW THE FILE IS WRITTEN
            weights = {key: w for key, w in zip(['42', '12', '21x21', '6x6'], json.loads(line[w_start_i:w_end_i+1]))}
            fold = int(line.strip()[-1])
            if fold == 1:
                results.append([])
            if weights['21x21'] == 0:
                results[-1].append(cur_res['6x6'])
            else:
                results[-1].append(cur_res['21x21'])
            if fold == 4:
                results[-1].append(weights)
    return results


if __name__ == '__main__':
    run_cross_val()
    # results = read_results("val_results.txt")
    # analyze_results(results)
