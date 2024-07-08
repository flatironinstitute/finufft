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
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print('stdout output:\n', e.stdout)
        print('stderr output:\n', e.stderr)
        print("Error executing command:", e)


# function that builds a string from a dictionary of arguments

def build_args(args):
    args_list = []
    for key, value in args.items():
        args_list.append(key + " " + value)
    return ' '.join(args_list)


# function

# example command to run:
# nsys profile -o cuperftest_profile ./cuperftest --prec f --n_runs 10 --method 1 --N1 256 --N2 256 --N3 256 --M 1E8 --tol 1E-6
# example arguments
args = {"--prec": "f",
        "--n_runs": "1",
        "--method": "1",
        "--N1": "256",
        # "--N2": "256",
        # "--N3": "256",
        "--M": "1E8",
        "--tol": "1E-6"}
# iterate over tol from 1E-6 to 1E-1
data = {
    'method': [],
    'throughput': [],
    'tolerance': []
}
for i in range(1, 7):
    args["--tol"] = "1E-" + str(i)
    print("Running with tol = 1E-" + str(i))
    for method in ['2', '1']:
        if method == '0':
            data['method'].append('auto')
        elif method == '1':
            data['method'].append('GM')
        elif method == '2':
            data['method'].append('SM')
        print("Method " + data['method'][-1])
        cmd = ["profile", "--force-overwrite", "true", "-o", "cuperftest_profile", cwd + "/cuperftest", build_args(args)]
        stdout = run_command("nsys", cmd)
        # skip all lines starting with # in stdout
        stdout = [x for x in stdout.splitlines() if not x.startswith("#")][:7]
        stdout = '\n'.join(stdout)
        # convert stdout to a dataframe from csv string
        dt = pd.read_csv(io.StringIO(stdout), sep=',')
        setpts = dt[dt["event"].str.contains("setpts")]['nupts/s'].sum()
        exec = dt[dt["event"].str.contains("exec")]['nupts/s'].sum()
        print(f'setpts pts/s: {setpts}')
        print(f'exec pts/s: {exec}')
        cmd = ["stats", "--force-overwrite=true", "--force-export=true", "--report", "cuda_gpu_trace", "--report", "cuda_gpu_kern_sum", "cuperftest_profile.nsys-rep",
               "--format=csv", "--output", "cuperftest"]
        stdout = run_command("nsys", cmd)
        # print(csv)
        dt = pd.read_csv("./cuperftest_cuda_gpu_trace.csv")
        # print(dt)
        # sum the "Total Time" column of the ones that contain "fft" in name
        # print(dt[dt["Name"].str.contains("fft") & ~dt["Name"].str.contains("cufinufft")])
        total_fft = dt[dt["Name"].str.contains("fft") & ~dt["Name"].str.contains("cufinufft")]['Duration (ns)'].sum()
        print(f'total_fft: {total_fft}')
        # drop all the rows with spread not in "Name"
        dt = dt[dt["Name"].str.contains("cufinufft::spreadinterp::spread")]
        # print(dt)
        # sort dt by column "Time (%)"
        total_spread = dt['Duration (ns)'].sum() - total_fft
        print(f'total_spread: {total_spread}')
        # pt/s
        throughput = float(args['--M']) * float(args['--n_runs']) * 1_000_000_000 / total_spread
        print(f'throughput: {throughput}')
        data['throughput'].append(throughput)
        data['tolerance'].append(args['--tol'])

df = pd.DataFrame(data)
print(df)
# Pivot the DataFrame
pivot_df = df.pivot(index='tolerance', columns='method', values='throughput')
# Plot
pivot_df.plot(kind='bar', figsize=(10, 7))
# Find the minimum throughput value
min_throughput = df['throughput'].min()

# Calculate the smallest power of 10
min_pow_10 = 10 ** np.floor(np.log10(min_throughput))

# Adjust the plot's y-axis limits
plt.ylim(df['throughput'].min()*.99, df['throughput'].max() * 1.09)  # Adding 10% for upper margin

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