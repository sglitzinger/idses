'''
Computes an integrated crown schedule as well as a device design (number of cores of each type) from a task set, a given deadline and a chip area budget, which may be set via the command line.
More than two core types are possible.
'''


# pweight, cores, MIPGap, core/area constraints, objective function


import gurobipy as grb
import random
import math
import pandas as pd
import sys
import os
import time
from enum import Enum

# Class that enumerates all the possible task types
class TaskType(Enum):
    MEMORY = 2
    BRANCH = 3
    FMULT = 6
    SIMD = 8
    MATMUL = 9

# freqs = [1.2, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 2.7, 2.9, 3.0, 3.2, 3.3, 3.5]
# Possible core frequencies of ARM big.LITTLE (1.6 GHz on big cores is ignored here)
freqs = [
    # big
    [0.6, 0.8, 1.0, 1.2, 1.4],
    # LITTLE
    [0.6, 0.8, 1.0, 1.2, 1.4],
    # A72
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
]
# freqs = [600, 800, 1000, 1200, 1400]
# Execution time on big core relative to execution time on LITTLE core for different task types
# in order: MEMORY, BRANCH, FMULT, SIMD, MATMUL, DEFAULT
execution_time_multipliers = [
    [0.676, 1.376, 0.264, 0.278, 0.746, 0.6677],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.670, 0.643, 0.132, 0.140, 0.232, 0.363]
]
# Per core power values for different core types, task types and frequency levels
powers = [
    # big cores
    [
        [0.265,0.3075,0.4625,0.5475,0.835,1.3625], # MEMORY
        [0.1025,0.1475,0.2575,0.3175,0.4875,0.8825],  # BRANCH
        [0.1225,0.1725,0.3025,0.3625,0.5575,0.9525],  # FMULT
        [0.3075,0.4225,0.76,0.8775,1.4075,2.455],  # SIMD
        [0.205,0.285,0.4775,0.585,0.885,1.5225],  # MATMUL
        [0.2005,0.267,0.452,0.538,0.8345,1.435]  # DEFAULT
    ],
    # LITTLE cores
    [
        [0.195,0.2325,0.24,0.2875,0.3075], # MEMORY
        [0.0775,0.1075,0.1275,0.195,0.2275],  # BRANCH
        [0.045,0.0675,0.0975,0.13,0.1475],  # FMULT
        [0.0775,0.1225,0.1475,0.2325,0.2675],  # SIMD
        [0.0725,0.1175,0.1425,0.2225,0.2575],  # MATMUL
        [0.0935,0.1295,0.151,0.2135,0.2415]  # DEFAULT
    ],
    # A72 cores
    [
        [0.176,0.233,0.267,0.299,0.332,0.365,0.396,0.436,0.461,0.493],
        [0.153,0.206,0.232,0.261,0.288,0.318,0.350,0.378,0.410,0.442],
        [0.215,0.277,0.318,0.357,0.399,0.441,0.484,0.528,0.572,0.616],
        [0.211,0.270,0.306,0.344,0.383,0.423,0.463,0.503,0.545,0.589],
        [0.316,0.388,0.441,0.502,0.560,0.618,0.676,0.732,0.794,0.828],
        [0.214,0.275,0.313,0.352,0.393,0.433,0.474,0.515,0.556,0.594]
    ]
]

base_powers = [0.745, 0.645, 0.595]
zeta = 1.0 # 0.649
kappa = 52.64
eta = 0.5
alpha = 3.0
deadline_factor = 1.3 # 1.0 # 0.6 # Adjust deadline
parallel_efficiencies = [1,0.9,0.88,0.86,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001]

areas = [19, 3.8, 53.63]
chip_area = 91.2
delta = 0.0 # Fraction of chip area not usable
c = 5
pweight = 0 # 0.001 # 0.005 # 0.05


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
    return 5


def get_power(coretype, tasktype, freqlevel):
    tt_index = get_tasktype_index(tasktype)
    if freqlevel < len(powers[coretype][tt_index]):
        return powers[coretype][tt_index][freqlevel]
    else:
        return 10e9


def par_eff(tau, psi, max_width, procs_alloc):
    if procs_alloc == 1:
        return 1.0
    elif procs_alloc == 2:
        return 0.9
    elif procs_alloc == 3:
        return 0.88
    elif procs_alloc == 4:
        return 0.86
    elif procs_alloc > 4:
        return 0.000001
    else:
        print("Allocation invalid!")
    #return parallel_efficiencies[int(procs_alloc)-1]


def core_eff(tasktype, coretype):
    return execution_time_multipliers[coretype][get_tasktype_index(tasktype)]


def create_task_data(dataframe, cores):
    workloads = dataframe['workload'].tolist()
    tasktypes = dataframe['tasktype'].tolist()
    max_widths = dataframe['max_width'].tolist()
    # Cap max width at half of total cores (no concurrent execution of tasks on big *and* LITTLE cores)
    for max_width in max_widths:
        if max_width > cores // 2:
            max_width = cores // 2
    psi_values = [0] * len(dataframe.index)
    return workloads, tasktypes, max_widths, psi_values


# Returns width of group
def get_group_width(group, procs):
    return procs / 2**math.floor(math.log(group, 2))


# Returns list of proc's groups
def get_groups(proc, procs):
    groups = []
    group = procs + proc - 1
    while group > 0:
        groups.append(group)
        group = group // 2
    groups.reverse()
    return groups


# Returns list of groups too large for task
def get_groups_too_large(task, max_width, groups, procs):
    grps_too_large = []
    for i in range(1, groups+1):
        if get_group_width(i, procs) > max_width:
            grps_too_large.append(i)
    return grps_too_large


# For details cf. Kessler et al. (2013)
def compute_deadline(df, cores):
    total_workload = df['workload'].sum()
    print("Total workload:", total_workload)
    # Account for potential parallel execution
    # total_workload /= 0.7
    # Account for faster execution on big cores
    # total_workload *= 0.83385
    # print("Total workload adjusted:", total_workload)

    ##########################################################################
    # CAREFUL: DEADLINE COMPUTATION RELIES ON FREQUENCIES FOR LITTLE CORES...#
    ##########################################################################
    lower_bound = total_workload / (cores * freqs[1][len(freqs[1])-1])
    print("Lower bound:", lower_bound)
    # upper_target = 2 * total_workload / (cores * freqs[0])
    upper_target = total_workload / (cores * freqs[1][0])
    # upper_target = lower_bound
    print("Upper target:", upper_target)
    return (lower_bound + upper_target) / 2


# Returns list of processors in group
def get_procs_in_group(group, cores):
    if group >= cores:
        return [group-cores+1]
    else:
        procs_in_group = []
        for i in range(1,cores+1):
            if group in get_groups(i, cores):
                procs_in_group.append(i)
        return procs_in_group


# Compute smallest core index amongst cores in group
def get_min_index(group, cores):
    cores_in_group = get_procs_in_group(group, cores)
    return min(cores_in_group)


# Compute largest core index amongst cores in group
def get_max_index(group, cores):
    cores_in_group = get_procs_in_group(group, cores)
    return max(cores_in_group)


# Compute chip size from number of big and LITTLE cores
def get_chip_size(numbig, numlittle, numa72):
    return numbig * areas[0] + numlittle * areas[1] + numa72 * areas[2]

# Arguments to be passed: path to task set file, path to results directory, task set file, max. chip size
def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    task_filename = sys.argv[3]
    global chip_area
    chip_area = float(sys.argv[4])

    # Read from info file
    # with open(os.path.join(output_path, info_filename), 'r') as infofile:
        # cores = int(infofile.readline())
        # deadline = float(infofile.readline())
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    results_filename = os.path.join(output_path, "results_dssched_biglittle.csv")
    optstat_filename = os.path.join(output_path, "optstat_dssched_biglittle.csv")
    
    maxcores = math.ceil((chip_area * (1-delta)) / min(areas))
    # Number of cores is smallest power of 2 >= maxcores
    # cores = 1<<(maxcores-1).bit_length()
    cores = 32

    df = pd.read_csv(os.path.join(input_path, task_filename), delimiter=',', usecols=["workload","max_width","tasktype"])

    cores_for_deadline = 8
    deadline = compute_deadline(df, cores_for_deadline) * deadline_factor
    print("Cores:", cores)
    print("Deadline:", deadline)

    num_tasks = len(df.index)
    num_groups = cores * 2 - 1
    num_types = 3 # 2 types, 0 is big, 1 is LITTLE
    num_freqs = []

    set_I = range(1, num_groups+1)
    set_J = range(0, num_tasks)
    set_T = range(0, num_types)
    for t in set_T:
        num_freqs.append(len(freqs[t]))
    maxnumfreqs = max([len(freqs[t]) for t in set_T])
    set_K = range(0, maxnumfreqs)
    
    set_M = range(1, cores+1)

    workloads, tasktypes, max_widths, psi_values = create_task_data(df, cores)

    opt_model = grb.Model(name="MIP Model")
    opt_model.setParam('TimeLimit', 5*60)
    #opt_model.setParam('MIPGap', 1e-6)

    x_vars = {(i,j,k,t):opt_model.addVar(vtype=grb.GRB.BINARY, name="x_{0}_{1}_{2}_{3}".format(i,j,k,t)) for i in set_I for j in set_J for k in set_K for t in set_T}
    # Number of cores of each type
    p_vars = {(t):opt_model.addVar(vtype=grb.GRB.INTEGER, name="p_{0}".format(t)) for t in set_T}
    # ex_i,t = 1 iff group i is not wholely composed of cores of type t and can therefore not be used for cores of this type when scheduling
    ex_vars = {(i,t):opt_model.addVar(vtype=grb.GRB.BINARY, name="ex_{0}_{1}".format(i,t)) for i in set_I for t in set_T}

    # Constraint: map each task exactly once
    for j in set_J:
        opt_model.addConstr(
            lhs=grb.quicksum(x_vars[i,j,k,t] for i in set_I for t in set_T for k in set_K if k < num_freqs[t]),
            sense=grb.GRB.EQUAL,
            rhs=1, 
            name="constraint_mapped_once_{0}".format(j)
        )
    for j in set_J:
        opt_model.addConstr(
            lhs=grb.quicksum(x_vars[i,j,k,t] for i in set_I for t in set_T for k in set_K if k >= num_freqs[t]),
            sense=grb.GRB.EQUAL,
            rhs=0, 
            name="constraint_not_mapped_{0}".format(j)
        )

    # Constraint: do not allocate a number of cores greater than maximum width of task
    for j in set_J:
        opt_model.addConstr(
            lhs=grb.quicksum(x_vars[i,j,k,t] for i in get_groups_too_large(j, max_widths[j], num_groups, cores) for k in set_K for t in set_T),
            sense=grb.GRB.EQUAL,
            rhs=0,
            name="constraint_group_size_{0}".format(j)
        )

    # Constraint: sum of task runtimes per core must not exceed deadline
    for t in set_T:
        for m in set_M:
            opt_model.addConstr(
                #lhs=grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], i)) / (get_group_width(i, cores) * freqs[k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in get_groups(m, cores) for j in set_J for k in set_K for t in set_T),
                lhs=grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], t)) / (get_group_width(i, cores) * freqs[t][k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in get_groups(m, cores) for j in set_J for k in set_K if k < num_freqs[t]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=deadline,
                name="constraint_deadline_{0}".format(m)
            )

    # Constraint: Number of cores of each type must not lead to excessive chip area usage
    opt_model.addConstr(
        lhs=grb.quicksum(p_vars[t] * areas[t] for t in set_T),
        sense=grb.GRB.LESS_EQUAL,
        rhs=chip_area * (1-delta), 
        name="constraint_chip_area"
    )

    ################################################################
    # MAPPING CONSTRAINTS TYPES
    ################################################################
    
    # Constraint: force ex_i,t to 1 if max(i) >= p_t
    for t in set_T:
        for i in set_I:
            opt_model.addConstr(
                lhs=get_max_index(i, cores) - ex_vars[i,t] * cores,
                sense=grb.GRB.LESS_EQUAL,
                rhs=p_vars[t],
                name="constraint_p_ex_{0}_{1}".format(t,i)
            )

    # Constraint: force ex_i,t to 0 if max(i) < p_t
    for t in set_T:
        for i in set_I:
            opt_model.addConstr(
                lhs=get_max_index(i, cores) + (1 - ex_vars[i,t]) * cores,
                sense=grb.GRB.GREATER_EQUAL,
                rhs=p_vars[t] + 1,
                name="constraint_pb_notex_{0}_{1}".format(t,i)
            )
    
    # Constraint: do not allow scheduling to groups with ex_i,b = 1
    for t in set_T:
        for i in set_I:
            for j in set_J:
                opt_model.addConstr(
                    lhs=grb.quicksum(x_vars[i,j,k,t] for k in set_K),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=1 - ex_vars[i,t],
                    name="constraint_pb_sched_{0}".format(t,i,j)
                )

    # Objective function: energy

    # No base power, no core minimization term
    #objective = grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], t) * get_power(t, tasktypes[j], k)) / (freqs[t][k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in set_I for j in set_J for k in set_K for t in set_T)
    
    # No base power, core minimization term
    #objective = grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], t) * get_power(t, tasktypes[j], k)) / (freqs[t][k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in set_I for j in set_J for k in set_K for t in set_T) + pweight * grb.quicksum(p_vars[t] for t in set_T)
    
    # Base power, no core minimization term
    objective = grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], t) * get_power(t, tasktypes[j], k)) / (freqs[t][k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in set_I for j in set_J for t in set_T for k in set_K if k < num_freqs[t]) + grb.quicksum(p_vars[t] * base_powers[t] * deadline for t in set_T)
    
    # Base power, core minimization (should deliver same results as base power, no core minimization term)
    #objective = grb.quicksum(x_vars[i,j,k,t] * ((workloads[j] * core_eff(tasktypes[j], t) * get_power(t, tasktypes[j], k)) / (freqs[t][k] * par_eff(workloads[j], psi_values[j], max_widths[j], get_group_width(i, cores)))) for i in set_I for j in set_J for k in set_K for t in set_T) + grb.quicksum(p_vars[t] * base_powers[t] * deadline for t in set_T) + pweight * grb.quicksum(p_vars[t] for t in set_T)

    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)

    opt_model.optimize()

    if opt_model.SolCount >= 1:
        opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])

        opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["group", "task", "frequency", "coretype"])

        opt_df.reset_index(inplace=True)

        opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.X)

        opt_df.drop(columns=["variable_object"], inplace=True)

        # for v in opt_model.getVars():
        #     print('%s %g' % (v.varName, v.x))

        # opt_df_three = opt_df[opt_df.task == 3]
        # opt_df_three.to_csv("./optdftest", index=False)

        opt_df_exp = opt_df[opt_df.solution_value > 0.5]
        # opt_df_exp.drop(columns=["solution_value"], inplace=True)

        #print(opt_df_exp)

        numbig = opt_model.getVarByName("p_0").X
        numlittle = opt_model.getVarByName("p_1").X
        numa72 = opt_model.getVarByName("p_2").X
        print("Number of big cores:", numbig)
        print("Number of LITTLE cores:", numlittle)
        print("Number of A72 cores:", numa72)

        ex_df = pd.DataFrame.from_dict(ex_vars, orient="index", columns = ["variable_object"])
        ex_df.index = pd.MultiIndex.from_tuples(ex_df.index, names=["group", "coretype"])
        ex_df.reset_index(inplace=True)
        ex_df["solution_value"] = ex_df["variable_object"].apply(lambda item: item.X)
        ex_df.drop(columns=["variable_object"], inplace=True)
        ex_df_exp = ex_df[ex_df.solution_value > 0.5]
        #print(ex_df_exp)

        filename = os.path.splitext(task_filename)[0] + "_dssched_alloc.csv"
        opt_df_exp.to_csv(os.path.join(output_path, filename), index=False, columns=["task", "group", "frequency", "coretype"])

        ############################################################################################################
        # CAREFUL: DO NOT FORGET TO SET VALUE FOR PWEIGHT WHEN EMPLOYING CORE MINIMIZATION IN OBJECTIVE FUNCTION!!!#
        ############################################################################################################
        energy = opt_model.objVal - pweight * (numbig + numlittle + numa72)

        with open(results_filename, 'a') as energyfile:
            if os.path.getsize(results_filename) == 0:
                energyfile.write("taskset,deadline,max. chip size,actual chip size,#big,#little,#a72,energy_consumption,objective function value\n")
            energyfile.write(os.path.splitext(task_filename)[0] + "," + str(deadline) + "," + str(chip_area) + "," + str(get_chip_size(numbig, numlittle, numa72)) + "," + str(numbig) + "," + str(numlittle) + "," + str(numa72) + "," + str(energy) + "," + str(opt_model.objVal) + "\n")

        print('Obj: %g' % opt_model.objVal)
        print("Energy consumption:", energy)
        print("MIPGap:", opt_model.MIPGap)
    else:
        print("No feasible solution found!")
        with open(results_filename, 'a') as energyfile:
            if os.path.getsize(results_filename) == 0:
                energyfile.write("taskset,deadline,chip size,#big,#little,#a72,energy_consumption,objective function value\n")
            energyfile.write(os.path.splitext(task_filename)[0] + "," + str(deadline) + "," + str(chip_area) + "," + "\n")
    with open(optstat_filename, 'a') as optstatfile:
        if os.path.getsize(optstat_filename) == 0:
            optstatfile.write("taskset,optimization status,MIPGap\n")
        optstatfile.write(os.path.splitext(task_filename)[0] + "," + str(opt_model.Status) + "," + str(opt_model.MIPGap) + "\n")
        

if __name__ == "__main__":
    start_time = time.process_time()
    main()
    with open("timesdssched.log", 'a+') as ctlog:
        ctlog.write(str(time.process_time() - start_time) + "\n")
