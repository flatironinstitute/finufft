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
        print("Running command:", ' '.join(cmd))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print('stdout output:\n', e.stdout)
        print('stderr output:\n', e.stderr)
        print("Error executing command:", e)


# function that builds a string from a dictionary of arguments

def build_args(args):
    args_list = []
    for key, value in args.items():
        args_list.append(key)
        args_list.append(value)
    return args_list


# function

# example command to run:
# nsys profile -o cuperftest_profile ./cuperftest --prec f --n_runs 10 --method 1 --N1 256 --N2 256 --N3 256 --M 1E8 --tol 1E-6
# example arguments
args = {"--prec": "f",
        "--n_runs": "5",
        "--method": "0",
        "--sort": "1",
        "--N1": "16777216",
        # "--N1": "256",
        # "--N2": "256",
        # "--N3": "256",
        "--kerevalmethod": "1",
        "--M": "1E8",
        "--tol": "1E-6"}
# iterate over tol from 1E-6 to 1E-1

warmup = {"--prec": "f",
        "--n_runs": "1",
        "--method": "0",
        "--N1": "256",
        # "--N2": "256",
        # "--N3": "256",
        "--M": "256",
        "--tol": "1E-1"}
cmd = ["profile", "--force-overwrite", "true", "-o", "cuperftest_profile", cwd + "/cuperftest"] + build_args(warmup)
print("Warmup")
stdout, stderr = run_command("nsys", cmd)
print("Benchmarking")
if stderr != '':
    print(stderr)
    exit(0)
for precision in ['d']:
    print(f"precision: {precision}")
    for dim in range(1, 2):
        if dim == 1:
            args["--N1"] = "16777216"
        if dim == 2:
            args["--N1"] = "256"
            args["--N2"] = "256"
        if dim == 3:
            args["--N1"] = "256"
            args["--N2"] = "256"
            args["--N3"] = "256"
        args["--prec"] = precision
        max_range = 16 if args["--prec"] == "d" else 7
        if precision == 'd' and dim == 3:
            max_range = 6
        print(f"dimensions {dim}")
        data = {
            'method': [],
            'throughput': [],
            'tolerance': [],
            # 'setpts': [],
            'exec': [],
        }
        for i in range(1, max_range):
            args["--tol"] = "1E-" + ("0" if i < 10 else "") + str(i)
            print("Running with tol = 1E-" + str(i))
            for method in ['2', '1']:
                args["--method"] = method
                if method == '0':
                    data['method'].append('auto')
                elif method == '1':
                    data['method'].append('GM')
                elif method == '2':
                    data['method'].append('SM')
                elif method == '4':
                    data['method'].append('BLOCK')
                print("Method " + data['method'][-1])
                cmd = ["profile", "--force-overwrite", "true", "-o", "cuperftest_profile", cwd + "/cuperftest"] + build_args(args)
                stdout, stderr = run_command("nsys", cmd)
                if stderr != '':
                    print(stderr)
                    exit(0)
                # skip all lines starting with # in stdout
                conf = [x for x in stdout.splitlines() if x.startswith("#")]
                print('\n'.join(conf))
                stdout = [x for x in stdout.splitlines() if not x.startswith("#")][:7]
                if stdout[0].startswith("bin"):
                    print(stdout[0])
                    stdout = stdout[1:]

                stdout = '\n'.join(stdout)
                # convert stdout to a dataframe from csv string
                dt = pd.read_csv(io.StringIO(stdout), sep=',')
                setpts = dt[dt["event"].str.contains("setpts")]['nupts/s'].sum() # it is only one row it extracts the value
                exec = dt[dt["event"].str.contains("exec")]['nupts/s'].sum() # it is only one row it extracts the value
                # print(f'setpts pts/s: {setpts}')
                # print(f'exec pts/s: {exec}')
                cmd = ["stats", "--force-overwrite=true", "--force-export=true", "--report", "cuda_gpu_trace", "--report", "cuda_gpu_kern_sum", "cuperftest_profile.nsys-rep",
                       "--format=csv", "--output", "cuperftest"]
                stdout, _ = run_command("nsys", cmd)
                # remove format from cmd
                cmd = cmd[:-3]
                # print(run_command("nsys", cmd))
                # print(csv)
                dt = pd.read_csv("./cuperftest_cuda_gpu_trace.csv")
                # print(dt)
                # sum the "Total Time" column of the ones that contain "fft" in name
                # print(dt[dt["Name"].str.contains("fft") & ~dt["Name"].str.contains("cufinufft")])
                total_fft = dt[dt["Name"].str.contains("fft") & ~dt["Name"].str.contains("cufinufft")]['Duration (ns)'].sum()
                # print(f'total_fft: {total_fft}')
                # drop all the rows with spread not in "Name"
                dt = dt[dt["Name"].str.contains("cufinufft::spreadinterp::spread")]
                # print(dt)
                # exit(0)
                # sort dt by column "Time (%)"
                total_spread = dt['Duration (ns)'].sum() - total_fft
                # print(f'total_spread: {total_spread}')
                if total_fft > total_spread:
                    print("Warning: total_fft > total_spread")
                    # exit(0)
                # pt/s
                throughput = float(args['--M']) * float(args['--n_runs']) * 1_000_000_000 / total_spread
                print(f'throughput: {throughput}')
                data['throughput'].append(throughput)
                data['tolerance'].append(args['--tol'])
                # data['setpts'].append(setpts)
                data['exec'].append(exec)
        df = pd.DataFrame(data)
        # Pivot the DataFrame
        pivot_df = df.pivot(index='tolerance', columns='method')
        # print(pivot_df)
        # scale the throughput SM by GM
        # pivot_df['throughput', 'SM'] /= pivot_df['throughput', 'GM']
        # pivot_df['throughput', 'GM'] /= pivot_df['throughput', 'GM']
        # scale setpts SM by GM
        # pivot_df['exec', 'SM'] /= pivot_df['exec', 'GM']
        # pivot_df['exec', 'GM'] /= pivot_df['exec', 'GM']
        # remove the GM column
        # pivot_df.drop(('throughput', 'GM'), axis=1, inplace=True)
        pivot_df.drop(('exec', 'GM'), axis=1, inplace=True)
        pivot_df.drop(('exec', 'SM'), axis=1, inplace=True)
        print(pivot_df)
exit(0)
# Plot
pivot_df.plot(kind='bar', figsize=(10, 7))
# Find the minimum throughput value
min_val = min(pivot_df[('throughput', 'SM')].min(), pivot_df[('throughput', 'GM')].min())
max_val = max(pivot_df[('throughput', 'SM')].max(), pivot_df[('throughput', 'GM')].max())
print(min_val, max_val)
plt.ylim(min_val * .90, max_val * 1.1)
# plt.ylim(.8, 1.2)

# Calculate the smallest power of 10
# min_pow_10 = 10 ** np.floor(np.log10(min_throughput))

# Adjust the plot's y-axis limits
# plt.ylim(df['throughput'].min()*.99, df['throughput'].max() * 1.009)  # Adding 10% for upper margin

# plot an horizontal line at 1 with label "GM"
# plt.axhline(y=1, color='k', linestyle='--', label='GM')
plt.xlabel('Tolerance')
plt.ylabel('Throughput')
plt.title('Throughput by Tolerance and Method')
plt.legend(title='Method')
plt.tight_layout()
plt.show()
plt.xlabel("Tolerance")
plt.ylabel("Points/s")
plt.savefig("bench.png")
plt.savefig("bench.svg")
plt.savefig("bench.pdf")
plt.show()
