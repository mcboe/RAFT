import subprocess
import numpy as np
import csv
import json
import pickle
from ruamel.yaml import YAML
from deap import base, creator, tools, algorithms
from copy import deepcopy

import multiprocessing
import uuid
import tempfile
import shutil
import os

n_threads = multiprocessing.cpu_count()
print(f"Number of threads (logical cores): {n_threads}")
# ----- File paths -----
yaml_path = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\TLP15MW-RAFT_QTFtest.yaml"
raft_script1 = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\example-RAFT_QTFTLP15MW.py"
raft_script2 = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\example-RAFT_QTFTLP15MW2.py"
raft_script3 = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\example-RAFT_QTFTLP15MW3.py"

results_json_path = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\case_resultsopt.pkl"
log_csv_path = "GA_raft_log.csv"

# Clear the CSV log at the start of the script
open(log_csv_path, "w").close()

with open(results_json_path, "rb") as f:
            results = pickle.load(f)


# ----- YAML handler -----
yaml = YAML()
yaml.preserve_quotes = True

# ----- Constraints -----
limits = {
    "surge": 15.0,
    "pitch": 10.0,
    "T_max": 30000000.0,
    "T_min": 1,
    "acc_nacelle": 2.5,
    "Fatigue_damage": 1
}

# ----- Log header -----
CSV_HEADER = ["d", "draft", "T_pre", "alpha", "L_pontoon", "D_pontoon", "weight", "surge", "pitch", "T_max", "T_min", "acc_nacelle","Fatigue_damage", "LCOE", "penalty"]


# ----- Evaluation function -----
def evaluate(individual):
    d, draft, T_pre, alpha, L_pontoon, D_pontoon = individual
    import glob
    #draft = 45
    #T_pre = 14000000
    #alpha = 90
    #L_pontoon = 27
    #D_pontoon = 2.7
    print(' individual' , d, draft, T_pre, alpha, L_pontoon, D_pontoon)

    Energy = 0
    Fatigue = 0
    # Hard bounds check
    #if not (10 <= d <= 28.0):
    #    print(f"❌ Diameter {d:.3f} out of bounds")
    #    return 1e20,

    work_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]  # Short ID
    individual_yaml_path = os.path.join(work_dir, f"config_{unique_id}.yaml")

    # Modify YAML
    base_yaml_path = yaml_path
    with open(base_yaml_path, "r") as f:
        base_data = yaml.load(f)

    data = deepcopy(base_data)  # Per-individual safe copy

    
    Concept = data["settings"]["Conceptcounter"]
    
    for m in data["platform"]["members"]:
        if m.get("name") == "main_column":
            m["d"] = [d, d]
            m["t"] = d/100
            m["rA"][2] = -draft
            m["stations"][0] = -draft
            m["cap_stations"] = [-draft]
            m["cap_t"] = [d/100]
        elif m.get("name") == "pontoon":
            m["rA"][0] = d / 2
            m["rA"][2] = -draft + D_pontoon / 2
            m["rB"][0] = L_pontoon
            m["rB"][2] = -draft + D_pontoon / 2
            m["stations"][0] = d / 2
            m["stations"][1] = L_pontoon
            m["d"] = [D_pontoon, D_pontoon]
            m["t"] = D_pontoon/100
            m["rA"][0] = d / 2

    data["mooring"]["pretension"] = T_pre

    for p in data["mooring"]["points"]:
        if p.get("name") == "line1_anchor":
            p["location"][0] = float(L_pontoon + np.sqrt(((120-draft)/np.sin(alpha/180*np.pi))**2 - (120-draft)**2))
        elif p.get("name") == "line2_anchor":
            p["location"][0] = float(-L_pontoon - np.sqrt(((120-draft)/np.sin(alpha/180*np.pi))**2 - (120-draft)**2))
        elif p.get("name") == "line3_anchor":
            p["location"][1] = float(L_pontoon + np.sqrt(((120-draft)/np.sin(alpha/180*np.pi))**2 - (120-draft)**2))
        elif p.get("name") == "line4_anchor":
            p["location"][1] = float(-L_pontoon - np.sqrt(((120-draft)/np.sin(alpha/180*np.pi))**2 - (120-draft)**2))
        elif p.get("name") == "line1_vessel":
            p["location"][0] = L_pontoon 
            p["location"][2] = -draft
        elif p.get("name") == "line2_vessel":
            p["location"][0] = -L_pontoon
            p["location"][2] = -draft
        elif p.get("name") == "line3_vessel":
            p["location"][1] = L_pontoon 
            p["location"][2] = -draft
        elif p.get("name") == "line4_vessel":
            p["location"][1] = -L_pontoon 
            p["location"][2] = -draft
    
    for p in data["mooring"]["lines"]:
        if p.get("name") == "line1":
            p["length"] = float(((120-draft)/np.sin(alpha/180*np.pi)))
        elif p.get("name") == "line2":
            p["length"] = float(((120-draft)/np.sin(alpha/180*np.pi)))
        elif p.get("name") == "line3":
            p["length"] = float(((120-draft)/np.sin(alpha/180*np.pi)))
        elif p.get("name") == "line4":
            p["length"] = float(((120-draft)/np.sin(alpha/180*np.pi)))
    
    data["mooring"]["line_types"][0]["diameter"]  = float(np.sqrt(((T_pre*4/(480*10**6))/np.pi)))   
    data["mooring"]["line_types"][0]["stiffness"] = float(2.1*10**11 *  (T_pre*4/(480*10**6)))

    with open(individual_yaml_path, "w") as f:
        yaml.dump(data, f)

    # Run RAFT simulation
    try:
        subprocess.run(["python", raft_script2, individual_yaml_path], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for diameter {d}")
        return 1e20,  # heavy penalty
    except subprocess.CalledProcessError:
        print(f"❌ Simulation error for diameter {d}")
        return 1e20,

    results = []
    pattern = os.path.join(work_dir, f"case_resultsopt.pkl")
    files = sorted(glob.glob(pattern))

    # Read RAFT output (adjust to match your actual result format)
    try:
        for path in files:
            with open(path, "rb") as f:
                result = pickle.load(f)
                results.append(result)
    except Exception as e:
        print(f"❌ Could not read output for diameter {d}: {e}")
        return 1e20,

    print(results)
    Mtotal = results[0]['properties']['total mass']
    #Mtotal = results['total mass']
    Buoyancy = results[0]['properties']['buoyancy (pgV)'] 
    verticalpretension = results[0]['properties']['F_lines0'][2] # np.linalg.norm(results['properties']['F_lines0'][0:3])
    #print(pretension)

    Equilcheck = Mtotal*9.81 - verticalpretension - Buoyancy

    Mplatform = results[0]['properties']['shell mass']
    Mballast = results[0]['properties']['ballast mass'][0]
    print('masses')
    print(Mplatform)
    print(Mballast)
    print(Mtotal)
    print(verticalpretension)
    print(Buoyancy)
    weight = Mplatform + Mballast
    #print(weight)
    if Mballast < 0 or Mplatform < 0 or np.abs(Equilcheck) >= 10:
        print(f"❌ Not physical")
        return 1e20,

    fns = results[0]['eigen']['frequencies']
    for f in range(len(fns)):
        if  0.083 <= f <= 0.126 or 0.249 <= f <= 0.378:
            print(f"❌ In 1P or 3P")
            return 1e20,
    
     # Run RAFT simulation
    try:
        subprocess.run(["python", raft_script3, individual_yaml_path], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for diameter {d}")
        return 1e20,  # heavy penalty
    except subprocess.CalledProcessError:
        print(f"❌ Simulation error for diameter {d}")
        return 1e20,


    results = []
    pattern = f"case_resultsCase*Concept{Concept}.pkl"  # N = concept number or design ID
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"❌ No output files found for concept {Concept}")
        return 1e20,

    for path in files:
        try:
            with open(path, "rb") as f:
                results.append(pickle.load(f))
        except Exception as e:
            print(f"❌ Could not read {path}: {e}")
            return 1e20,

    for res in results:
        surge_avg = np.abs(res['surge_avg'])
        pitch_avg = np.abs(res['pitchHub_avg'])
        T_max_avg = res['Tmoor_avg'][1]
        T_min_avg = res['Tmoor_avg'][0]
        acc_avg = np.abs(res['AxRNA_avg'][0])
        if surge_avg>=limits["surge"]:
            print(f"❌ mean surge already exceeds limit")
            return 1e20,
        if pitch_avg>=limits["pitch"]:
            print(f"❌ mean pitch already exceeds limit")
            return 1e20,
        if T_max_avg>=limits["T_max"]:
            print(f"❌ mean max tension already exceeds limit")
            return 1e20,
        if T_min_avg<=limits["T_min"]:
            print(f"❌ mean min tension already exceeds limit")
            return 1e20,
        if acc_avg>=limits["acc_nacelle"]:
            print(f"❌ mean nacelle acceleration already exceeds limit")
            return 1e20,

    # Run RAFT simulation
    try:
        subprocess.run(["python", raft_script1, individual_yaml_path], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for diameter {d}")
        return 1e20,  # heavy penalty
    except subprocess.CalledProcessError:
        print(f"❌ Simulation error for diameter {d}")
        return 1e20,

    import glob

    results = []
    pattern = os.path.join(work_dir, f"case_resultsCase*Concept*.pkl")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"❌ No output files found for concept {Concept}")
        return 1e20,

    for path in files:
        try:
            with open(path, "rb") as f:
                results.append(pickle.load(f))
        except Exception as e:
            print(f"❌ Could not read {path}: {e}")
            return 1e20,

    def normalized_penalty(val, limit, kind='upper'):
        if kind == 'upper':
            return ((val - limit) / limit) ** 2 if val > limit else 0
        elif kind == 'lower':
            return ((limit - val) / limit) ** 2 if val < limit else 0

    penalty = 0
    for res in results:
        surge = res['surge_max']
        pitch = res['pitchHub_max']
        T_max = res['Tmoor_max'][1]
        T_min = res['Tmoor_min'][0]
        acc = res['AxRNA_max'][0]
        averagepower = res['power_avg']

        lifetime = 25*365*24*3600 #[s]
        Energy += averagepower*lifetime

        Fatigue += res['fatiguedamage']

        penalty += normalized_penalty(surge, limits["surge"], 'upper')
        penalty += normalized_penalty(pitch, limits["pitch"], 'upper')
        penalty += normalized_penalty(T_max, limits["T_max"], 'upper')
        penalty += normalized_penalty(T_min, limits["T_min"], 'lower')
        penalty += normalized_penalty(acc, limits["acc_nacelle"], 'upper')

        #Mplatform = res['shell mass']
        #Mballast = res['ballast mass'][0]
        

    penalty += normalized_penalty(Fatigue, limits["Fatigue_damage"], 'upper')

    # Simple platform weight model (e.g., proportional to d^2)
    
    print('masses')
    print(Mplatform)
    print(Mballast)
    print(verticalpretension)
    weight = Mplatform + Mballast

    capacityfactor = 0.55
    total_energy = (Energy*capacityfactor)/(1*10**6 * 3600)

    cost = 800*Mplatform+150*Mballast+25*verticalpretension/1000 +  4352139.3 + (50129.6*15 + 28.6*15 + 23.1*15)*25 + 137500*15 # [$]

    LCOE = cost/total_energy
    # Penalty calculation
    # penalty = 0
    # if surge > limits["surge"]: penalty += (surge - limits["surge"]) ** 2
    # if pitch > limits["pitch"]: penalty += (pitch - limits["pitch"]) ** 2
    # if T_max > limits["T_max"]: penalty += (T_max - limits["T_max"]) ** 2
    # if T_min < limits["T_min"]: penalty += (limits["T_min"] - T_min) ** 2
    # if acc > limits["acc_nacelle"]: penalty += (acc - limits["acc_nacelle"]) ** 2

    # Log run
    row = {
        "d": d, 
        "draft": draft, 
        "T_pre":T_pre , 
        "alpha": alpha, 
        "L_pontoon":L_pontoon , 
        "D_pontoon": D_pontoon,
        "weight": weight,
        "surge": surge,
        "pitch": pitch,
        "T_max": T_max,
        "T_min": T_min,
        "acc_nacelle": acc,
        "Fatigue_damage": Fatigue,
        "LCOE": LCOE,
        "penalty": penalty
    }

    with open(log_csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if f.tell() == 0:  # First write?
            writer.writeheader()
        writer.writerow(row)
    print(weight)
    print(penalty)
    fitness = float(LCOE) + 1e5 * float(penalty)

    # Modify YAML
    # with open(yaml_path, "r") as f:
    #     data = yaml.load(f)

    
    # #Concept = data["settings"]["Conceptcounter"] + 1
    # #data["settings"]["Conceptcounter"] =  Concept

    # with open(yaml_path, "w") as f:
    #     yaml.dump(data, f)

    shutil.rmtree(work_dir)

    return (fitness),  # Comma needed (tuple)

# ----- DEAP GA setup -----
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Variable bounds
#      Diameter, draft, pretension, angle, pontoon length, pontoon diameter
LOW =  [10     , 15   , 0         , 70   , 5             , 1  ]
UP  =  [20     , 110  , 30000000     , 90   , 60            , 28.28]

def init_individual():
    d = np.random.uniform(10, 20)  # Main diameter
    draft = np.random.uniform(15, 110)
    T_pre = np.random.uniform(0, 50000000)
    alpha = np.random.uniform(70, 90)
    L_pontoon = np.random.uniform(d/2, 60)
    D_pontoon = np.random.uniform(1, d/2 * np.sqrt(2))  # upper bound depends on 'd'
    #draft = 45
    #T_pre = 14000000
    #alpha = 90
    #L_pontoon = 27
    #D_pontoon = 2.7

    LOW =  [10     , 15   , 0         , 70   , d/2            , 1  ]
    UP  =  [20     , 110  , 30000000     , 90   , 60            ,  d/2 * np.sqrt(2)]

    return creator.Individual([d, draft, T_pre, alpha, L_pontoon, D_pontoon])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=LOW, up=UP, eta=15.0)

toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=LOW, up=UP, eta=20.0, indpb=1.0 / len(LOW))

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

print('test')

# ----- Run the GA -----
if __name__ == "__main__":
    # Modify YAML
    with open(yaml_path, "r") as f:
        data = yaml.load(f)

    #data["settings"]["Conceptcounter"] = 1

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    pop = toolbox.population(n=3)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pool = multiprocessing.Pool(processes=2)
    toolbox.register("map", pool.map)

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=3, stats=stats, halloffame=hof, verbose=True)

    print("\n✅ Best design found:")
    best = hof[0]
    print(f"  Genome: {best}")
    print(f"  Fitness: {best.fitness.values[0]}")
