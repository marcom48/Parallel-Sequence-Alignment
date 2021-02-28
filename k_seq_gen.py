import sys
import random
import time

random.seed(time.time())
k = int(sys.argv[1])
n = int(sys.argv[2])

bases = "ACGT"

with open(f"{k}-{n}.dat",'w') as file:
    file.write(f"3\n2\n{k}\n")
    for i in range(k):
        gene = ""
        for j in range(n):
            gene += random.choice(bases)
        file.write(gene + "\n")
