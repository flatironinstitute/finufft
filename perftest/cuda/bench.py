import matplotlib.pyplot as plt
import os
import subprocess
import pandas as pd
import numpy as np

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
        run_command("nsys", cmd)
        cmd = ["stats", "--force-overwrite=true", "--force-export=true", "--report", "cuda_gpu_trace", "--report", "cuda_gpu_kern_sum", "cuperftest_profile.nsys-rep",
               "--format=csv", "--output", "cuperftest"]
        csv = run_command("nsys", cmd)
        print(csv)
        dt = pd.read_csv("./cuperftest_cuda_gpu_kern_sum.csv")
        # sort dt by column "Time (%)"
        dt = dt[dt["Name"].str.contains("cufinufft::spreadinterp::spread")]
        dt = dt.sort_values(by="Time (%)", ascending=False)
        # drop all the rows with spread not in "Name"
        time = dt["Avg (ns)"].values[0]
        # pt/s
        throughput = float(args['--M']) * 1_000_000_000 / time
        data['throughput'].append(throughput)
        data['tolerance'].append(args['--tol'])

df = pd.DataFrame(data)

# Pivot the DataFrame
pivot_df = df.pivot(index='tolerance', columns='method', values='throughput')
# Plot
pivot_df.plot(kind='bar', figsize=(10, 7))
# Find the minimum throughput value
min_throughput = df['throughput'].min()

# Calculate the smallest power of 10
min_pow_10 = 10 ** np.floor(np.log10(min_throughput))

# Adjust the plot's y-axis limits
plt.ylim(df['throughput'].min()*.95, df['throughput'].max() * 1.05)  # Adding 10% for upper margin

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