import subprocess
import numpy as np
import csv
import json
import pickle
from ruamel.yaml import YAML
from deap import base, creator, tools, algorithms

# ----- File paths -----
yaml_path = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\TLP15MW-RAFT_QTFtest.yaml"
raft_script1 = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\example-RAFT_QTFTLP15MW.py"
raft_script2 = r"C:\Users\mcboe\OneDrive - Delft University of Technology\Documenten\Master ODE\Afstuderen\Github\RAFT\examples\example-RAFT_QTFTLP15MW2.py"

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
    "T_min": 0.0,
    "acc_nacelle": 2.5
}

# ----- Log header -----
CSV_HEADER = ["d", "weight", "surge", "pitch", "T_max", "T_min", "acc_nacelle", "penalty"]

def getdependencies(d):
    t = 0.01*d

# ----- Evaluation function -----
def evaluate(individual):
    d = individual[0]

    t = getdependencies(d)
    # Hard bounds check
    if not (14 <= d <= 28.0):
        print(f"❌ Diameter {d:.3f} out of bounds")
        return 1e9,

    # Modify YAML
    with open(yaml_path, "r") as f:
        data = yaml.load(f)

    for m in data["platform"]["members"]:
        if m.get("name") == "main_column":
            m["d"] = d
            m["t"] = d
        if m.get("name") == "pontoon":
            m["rA"][0] = d / 2
            m["stations"][0] = d / 2

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    # Run RAFT simulation
    try:
        subprocess.run(["python", raft_script2], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for diameter {d}")
        return 1e9,  # heavy penalty
    except subprocess.CalledProcessError:
        print(f"❌ Simulation error for diameter {d}")
        return 1e9,

    # Read RAFT output (adjust to match your actual result format)
    try:
        with open(results_json_path, "rb") as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"❌ Could not read output for diameter {d}: {e}")
        return 1e9,

    Mtotal = results['properties']['total mass']
    Buoyancy = results['properties']['buoyancy (pgV)'] 
    pretension = results['properties']['F_lines0'][2]

    Equilcheck = Mtotal*9.81 - pretension - Buoyancy

    Mplatform = results['properties']['shell mass']
    Mballast = results['properties']['ballast mass'][0]
    print('masses')
    print(Mplatform)
    print(Mballast)
    weight = Mplatform + Mballast
    print(weight)
    if Mballast < 0:
        print(f"❌ Not physical")
        return 1e9,


    # Run RAFT simulation
    try:
        subprocess.run(["python", raft_script1], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for diameter {d}")
        return 1e9,  # heavy penalty
    except subprocess.CalledProcessError:
        print(f"❌ Simulation error for diameter {d}")
        return 1e9,

    # Read RAFT output (adjust to match your actual result format)
    try:
        with open(results_json_path, "rb") as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"❌ Could not read output for diameter {d}: {e}")
        return 1e9,

    surge =results['case_metrics'][0][0]['surge_max']
    pitch = results['case_metrics'][0][0]['pitchHub_max']
    T_max = results['case_metrics'][0][0]['Tmoor_max'][1]
    T_min = results['case_metrics'][0][0]['Tmoor_min'][0]
    acc = results['case_metrics'][0][0]['AxRNA_max'][0]

    # Simple platform weight model (e.g., proportional to d^2)
    Mplatform = results['properties']['shell mass']
    Mballast = results['properties']['ballast mass'][0]
    print('masses')
    print(Mplatform)
    print(Mballast)
    weight = Mplatform + Mballast
    # Penalty calculation
    penalty = 0
    if surge > limits["surge"]: penalty += (surge - limits["surge"]) ** 2
    if pitch > limits["pitch"]: penalty += (pitch - limits["pitch"]) ** 2
    if T_max > limits["T_max"]: penalty += (T_max - limits["T_max"]) ** 2
    if T_min < limits["T_min"]: penalty += (limits["T_min"] - T_min) ** 2
    if acc > limits["acc_nacelle"]: penalty += (acc - limits["acc_nacelle"]) ** 2

    # Log run
    row = {
        "d": d,
        "weight": weight,
        "surge": surge,
        "pitch": pitch,
        "T_max": T_max,
        "T_min": T_min,
        "acc_nacelle": acc,
        "penalty": penalty
    }

    with open(log_csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if f.tell() == 0:  # First write?
            writer.writeheader()
        writer.writerow(row)
    print(weight)
    print(penalty)
    fitness = float(weight) + 1e5 * float(penalty)
    return (fitness),  # Comma needed (tuple)

# ----- DEAP GA setup -----
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_diameter", np.random.uniform, 17.0, 18.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_diameter, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=1.0)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# ----- Run the GA -----
if __name__ == "__main__":
    pop = toolbox.population(n=2)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=3, stats=stats, halloffame=hof, verbose=True)

    print("\n✅ Best design found:")
    print(f"  Diameter = {hof[0][0]:.2f} m")
