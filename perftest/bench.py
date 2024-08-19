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
fft_lib = ['fftw', 'ducc']
upsamp = ['1.25', '2.00']

# clone the repository
run_command('git', ['clone', 'https://github.com/DiamonDinoia/finufft.git'])

all_data = pd.DataFrame()

args = {'--prec': 'f',
        '--n_runs': '1',
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
    plt.legend()

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
        run_command('git', ['-C', 'finufft', 'checkout', 'origin/perftests', '--', 'perftest'])
        if fft == 'ducc':
            run_command('cmake', ['-S', 'finufft', '-B', 'build', '-DFINUFFT_BUILD_TESTS=ON', '-DFINUFFT_USE_DUCC0=ON'])
        else:
            run_command('cmake', ['-S', 'finufft', '-B', 'build', '-DFINUFFT_BUILD_TESTS=ON'])
        run_command('cmake', ['--build', 'build', '-j', str(os.cpu_count()), '--target', 'perftest'])

        for upsampfac in upsamp:
            args['--upsampfac'] = upsampfac
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
            dt['upsampfac'] = upsampfac
            # for key, value in args.items():
            #     dt[key[2:]] = value
            print(dt)
            all_data = pd.concat((all_data, dt), ignore_index=True)


print(all_data)

for upsampfac in upsamp:
    this_data = all_data[all_data['upsampfac'] == upsampfac]
    baseline = this_data[this_data['version'] == '2.2.0-fftw']
    baseline_amortized = baseline[baseline['event'] == 'amortized']['mean(ms)'].values[0]
    this_data['speedup'] = baseline_amortized / this_data[this_data['event'] == 'amortized']['mean(ms)']
    pivot = this_data.pivot(index='version', columns='event', values='mean(ms)')
    pivot = pivot.drop(columns='amortized')
    plot_stacked_bar_chart(pivot, this_data, args, '2d-' + str(upsampfac))

if __name__ == '__main__':
    pass
