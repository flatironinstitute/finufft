from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import subprocess
import pandas as pd
import numpy as np
import io
from numbers import Number
import time


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


# clone the repository
run_command('git', ['clone', 'https://github.com/flatironinstitute/finufft.git'])


def get_cpu_temperature():
    try:
        # Run the sensors command
        output, _ = run_command('sensors', [])
        # Parse the output to find the CPU temperature
        for line in output.split('\n'):
            if 'Core 0' in line:
                # Extract the temperature value
                temp_str = line.split()[2]
                return temp_str
    except subprocess.CalledProcessError as e:
        print('Error executing sensors command:', e)
        return None


all_data = pd.DataFrame()

args = {
    '--type': '1',
    '--prec': 'f',
    '--n_runs': '10',
    '--sort': '1',
    '--N1': '320',
    '--N2': '320',
    '--N3': '1',
    '--ntransf': '1',
    '--thread': '1',
    '--M': '1E6',
    '--tol': '1E-5',
    '--upsampfac': '1.25'
}


@dataclass
class Params:
    prec: str
    N1: Number
    N2: Number
    N3: Number
    ntransf: int
    thread: int
    M: Number
    tol: float


thread_num = int(os.cpu_count())

versions = ['v2.2.0', 'v2.3.0-rc1']
fft_lib = ['fftw', 'ducc']
upsamp = ['1.25', '2.00']
transform = ['1', '2', '3']

ParamList = [
    Params('f', 1e4, 1, 1, 1, 1, 1e7, 1e-4),
    Params('d', 1e4, 1, 1, 1, 1, 1e7, 1e-9),
    Params('f', 320, 320, 1, 1, 1, 1e7, 1e-5),
    Params('d', 320, 320, 1, 1, 1, 1e7, 1e-9),
    Params('f', 320, 320, 1, thread_num, thread_num, 1e7, 1e-5),
    Params('d', 192, 192, 128, 1, thread_num, 1e7, 1e-7),
]


def plot_stacked_bar_chart(pivot_data, speedup_data, args, figname):
    categories = list(pivot_data.keys())
    versions = list(pivot_data[categories[0]].keys())

    # Create a figure
    plt.figure(figsize=(7.5, 7.5))

    # Initialize the bottom array for stacking
    bottom = np.zeros(len(versions))

    # Plot each category
    for category in categories:
        values = [pivot_data[category][version] for version in versions]
        plt.bar(versions, values, bottom=bottom, label=category)
        bottom += np.array(values)

    # Add speedup annotations
    for i, version in enumerate(versions):
        speedup = \
            speedup_data[(speedup_data['version'] == version) & (speedup_data['event'] == 'amortized')][
                'speedup'].values[0]
        plt.text(i, bottom[i], f'{speedup:.2f}x', ha='center', va='bottom')

    # Add labels and title
    plt.ylabel('time (ms)')
    plt.xlabel('version')
    plt.xticks(rotation=0)

    # for txt create a string with all the arguments insert a newline after half of the arguments
    txt = ''
    for i, (key, value) in enumerate(args.items()):
        txt += f'{key[2:]}:{value}'
        if i == len(args) // 2:
            txt += '\n'
        else:
            txt += ' '

    plt.title(txt)
    plt.legend(loc='lower left')

    # Save the figure
    plt.savefig(figname + '.png')
    plt.show()


for version in versions:
    for fft in fft_lib:
        if version == 'v2.2.0' and fft == 'ducc':
            continue
        run_command('git', ['-C', 'finufft', 'reset', '--hard'])
        run_command('git', ['-C', 'finufft', 'checkout', version])
        # checkout folder perftest from master branch
        run_command('git', ['-C', 'finufft', 'checkout', 'origin/master', '--', 'perftest'])
        # run_command('rm', ['-rf', 'build'])
        if fft == 'ducc':
            run_command('cmake',
                        ['-S', 'finufft', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', '-DFINUFFT_BUILD_TESTS=ON',
                         '-DFINUFFT_USE_DUCC0=ON'])
        else:
            run_command('cmake',
                        ['-S', 'finufft', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', '-DFINUFFT_BUILD_TESTS=ON',
                         '-DFINUFFT_FFTW_LIBRARIES=DOWNLOAD'])
        run_command('cmake', ['--build', 'build', '-j', str(os.cpu_count()), '--target', 'perftest'])
        for param in ParamList:
            for key, value in param.__dict__.items():
                args['--' + key] = str(value)
            for type in transform:
                args['--' + 'type'] = type
                for upsampfac in upsamp:
                    args['--upsampfac'] = upsampfac
                    # while (cpu_temp := float(get_cpu_temperature()[:-2])) > 44.0:
                    #     print(f'CPU temperature is {cpu_temp}°C, waiting for it to cool down...')
                    #     time.sleep(5)
                    if param.thread == 1:
                        out, _ = run_command('taskset', ['-c', '0', 'build/perftest/perftest'] + build_args(args))
                    else:
                        time.sleep(30)
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
                    dt['version'] = version[1:] + '-' + fft
                    for key, value in args.items():
                        dt[key[2:]] = value
                    all_data = pd.concat((all_data, dt), ignore_index=True)
                    print(dt)

# Replace the amortized event in all_data
all_data.loc[all_data['event'] == 'amortized', 'min(ms)'] = (
        all_data[all_data['event'] == 'makeplan']['min(ms)'].values
        + all_data[all_data['event'] == 'setpts']['min(ms)'].values
        + all_data[all_data['event'] == 'execute']['min(ms)'].values
)

print(all_data)
all_data.to_csv('all_data.csv')
for param in ParamList:
    for key, value in param.__dict__.items():
        args['--' + key] = str(value)
    for upsampfac in upsamp:
        args['--upsampfac'] = upsampfac
        for type in transform:
            args['--' + 'type'] = type
            this_data = all_data
            # select data for this specific upsampling factor, type and parameters
            for key, value in args.items():
                this_data = this_data[this_data[key[2:]] == value]
            print(this_data)
            name = f'{int(param.N1)}x{int(param.N2)}x{int(param.N3)}-type-{type}-upsamp{upsampfac}-prec{param.prec}-thread{int(param.thread)}'
            # select the baseline data
            baseline = this_data[this_data['version'] == '2.2.0-fftw']
            # calculate the amortized time for the baseline
            baseline_amortized = baseline[baseline['event'] == 'amortized']['min(ms)'].values[0]
            # calculate the speedup for all the other versions
            this_data['speedup'] = baseline_amortized / this_data[this_data['event'] == 'amortized']['min(ms)']
            # pivot the data
            pivot = this_data.pivot(index='version', columns='event', values='min(ms)')
            pivot = pivot.drop(columns='amortized')
            # plot the stacked bar chart
            plot_stacked_bar_chart(pivot, this_data, args, name)

# this_data = all_data[all_data['upsampfac'] == upsampfac]
# baseline = this_data[this_data['version'] == '2.2.0-fftw']
# baseline_amortized = baseline[baseline['event'] == 'amortized']['mean(ms)'].values[0]
# this_data['speedup'] = baseline_amortized / this_data[this_data['event'] == 'amortized']['mean(ms)']
# pivot = this_data.pivot(index='version', columns='event', values='mean(ms)')
# pivot = pivot.drop(columns='amortized')
# plot_stacked_bar_chart(pivot, this_data, args, '2d-' + str(upsampfac))

if __name__ == '__main__':
    pass
