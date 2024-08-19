import matplotlib.pyplot as plt
import os
import subprocess
import pandas as pd
import numpy as np
import io
cwd = os.getcwd()


# function that runs a command line command and returns the output
# it also takes a list of arguments to pass to the command
def run_command(command, args):
    # convert command and args to a string
    try:
        cmd = [command] + args
        print('Running command:', ' '.join(cmd))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print('stdout output:\n', e.stdout)
        print('stderr output:\n', e.stderr)
        print('Error executing command:', e)

def build_args(args):
    args_list = []
    for key, value in args.items():
        args_list.append(key + '=' + value)
    return args_list

versions = ['2.2.0', 'master']

# clone the repository
run_command('git', ['clone', 'https://github.com/DiamonDinoia/finufft.git'])
run_command('git', ['-C','finufft', 'fetch'])
run_command('git', ['-C','finufft', 'checkout', 'v2.2.0'])
# checkout folder perftest from master branch
run_command('git', ['-C','finufft', 'checkout', 'perftests', '--', 'perftest'])
run_command('cmake', ['-S', 'finufft', '-B', 'build', '-DFINUFFT_BUILD_TESTS=ON'])
run_command('cmake', ['--build', 'build', '-j', str(os.cpu_count()), '--target', 'perftest'])
args = {'--prec': 'f',
        '--n_runs': '5',
        '--method': '0',
        '--sort': '1',
        '--N1': '16777216',
        # '--N1': '256',
        # '--N2': '256',
        # '--N3': '256',
        '--kerevalmethod': '1',
        '--M': '1E5',
        '--tol': '1E-6'}

print(run_command('build/perftest/perftest', build_args(args)))

if __name__ == '__main__':
    pass
