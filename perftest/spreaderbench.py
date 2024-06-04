fast = 'new.txt'
slow = 'old.txt'


def read_data(filename):
    data = [0] * 17
    with open(filename) as f1:
        nspread = 0
        speed = 0
        for line in f1:
            if 'nspread' in line:
                nspread = int(line.split('=')[-1])
            if 'pts/s' in line:
                speed = float(line.split(' ')[12])
            data[nspread] = speed
    return data

# compute relative increment in percentage between two numbers


vec = read_data(fast)[2:]
old = read_data(slow)[2:]

# 1 : slow = x : fast
# x = (1 - slow/fast) * 100
i = 2
for vec, old in zip(vec, old):
    diff = (1 - old/vec)*100
    print(f'nspread={i:02d} delta={diff:.3f}%')
    i+=1
