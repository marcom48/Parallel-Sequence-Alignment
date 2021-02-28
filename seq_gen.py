import sys
import random
import time

random.seed(time.time())
n = int(sys.argv[1])

bases = "ACGT"

with open(f"{n}.dat",'w') as file:
    file.write(f"3\n2\n")
    for i in range(2):
        gene = ""
        for j in range(n):
            gene += random.choice(bases)
        file.write(gene + "\n")
