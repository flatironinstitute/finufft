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

versions = ['v2.2.0', 'v2.3.0-rc1']
fft = ['fftw', 'ducc']

# clone the repository
run_command('git', ['clone', 'https://github.com/DiamonDinoia/finufft.git'])

all_data = pd.DataFrame()

for version in versions:
    run_command('git', ['-C','finufft', 'checkout', version])
    # checkout folder perftest from master branch
    run_command('git', ['-C','finufft', 'checkout', 'origin/perftests', '--', 'perftest'])
    run_command('cmake', ['-S', 'finufft', '-B', 'build', '-DFINUFFT_BUILD_TESTS=ON'])
    run_command('cmake', ['--build', 'build', '-j', str(os.cpu_count()), '--target', 'perftest'])
    args = {'--prec': 'f',
            '--n_runs': '1',
            '--sort': '1',
            '--N1': '320',
            '--N2': '320',
            '--N3': '1',
            '--ntransf': '1',
            '--thread': '1',
            '--M': '1E6',
            '--tol': '1E-5'}

    out, _ = run_command('build/perftest/perftest', build_args(args))

    # parse the output, escape all the lines that start with #
    out = io.StringIO(out)
    lines = out.readlines()
    conf = [line for line in lines if line.startswith('#')]
    print(*conf, sep='')
    stdout = '\n'.join([line for line in lines if not line.startswith('#')])
    # convert stdout to a dataframe from csv string
    dt = pd.read_csv(io.StringIO(stdout), sep=',')
    # add columns with version and configuration
    dt['version'] = version[1:]
    for key, value in args.items():
        dt[key[2:]] = value
    print(dt)
    all_data = pd.concat((all_data, dt), ignore_index=True)
    print(all_data)

if __name__ == '__main__':
    pass
