'''
Counts occurrences of task type/core type mappings within scheduling results.
'''


import pandas as pd


num_ttypes = 5
tasktypes = ["MEMORY", "BRANCH", "FMULT", "SIMD", "MATMUL"]
num_ctypes = 3
# Core types are:
# - 0: big
# - 1: LITTLE
# - 2: A72

taskcounts = [10,20,40,80]
num_ts = 10


def get_tasktype_index(tasktype):
    if tasktype == 'MEMORY':
        return 0
    elif tasktype == 'BRANCH':
        return 1
    elif tasktype == 'FMULT':
        return 2
    elif tasktype == 'SIMD':
        return 3
    elif tasktype == 'MATMUL':
        return 4
    raise ValueError("Task type unknown!")


mappings_count = [None] * num_ttypes
for i in range(len(mappings_count)):
    mappings_count[i] = [0] * num_ctypes

for count in taskcounts:
    for i in range(1,num_ts+1):
        resfile = "./results/" + "low_n" + str(count) + "_" + str(i) + "_biglittle" + "_dssched_alloc.csv"
        tsfile = "./tasksetsbiglittleadjusted/" + "low_n" + str(count) + "_" + str(i) + "_biglittle" + ".csv"

        df_res = pd.read_csv(resfile)
        df_ts = pd.read_csv(tsfile)

        for index, row in df_res.iterrows():
            task = row['task']
            ctype = row['coretype']
            ts_row = df_ts[df_ts['id'] == task]
            ttype = ts_row.iloc[0]['tasktype']
            mappings_count[get_tasktype_index(ttype)][ctype] += 1

with open("./mappings_count.csv", "w") as mcf:
    mcf.write("task type,# mapped to big,# mapped to LITTLE,# mapped to A72\n")
    for i in range(num_ttypes):
        mcf.write(tasktypes[i])
        for j in range(num_ctypes):
            mcf.write("," + str(mappings_count[i][j]))
        mcf.write("\n")
