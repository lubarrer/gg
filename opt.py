import os
import pickle
from rdkit import Chem
from gg.ga import Genetic
from gg.moves import Crossover, Mutate
from gg.score import PCEScore
from ax import ParameterType, RangeParameter, SearchSpace, SimpleExperiment, modelbridge

filename = os.path.join(os.getcwd(), "data.csv")
co = Crossover(n_tries=10)
mu = Mutate(rate=0.01, n_tries=10, size_mean=39.15, size_std=3.50)
model = pickle.load(open("model.p", "rb"))
xt = pickle.load(open("xt.p", "rb"))
yt = pickle.load(open("yt.p", "rb"))
sc = PCEScore(filename, model, xt, yt, sa_score=False, cycle_score=False)

axga_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="generations", parameter_type=ParameterType.INT, lower=10, upper=20
        ),
        RangeParameter(
            name="population_size", parameter_type=ParameterType.INT, lower=10, upper=20
        ),
        RangeParameter(
            name="mating_pool_size",
            parameter_type=ParameterType.INT,
            lower=10,
            upper=20,
        ),
    ]
)


def axga(generations, population_size, mating_pool_size):
    params = {
        "generations": generations,
        "population_size": population_size,
        "mating_pool_size": mating_pool_size,
        "prune_population": True,
        "max_score": 20.0,
        "filename": filename,
    }
    ga = Genetic(co, mu, sc, params)
    scores, population, generation = ga()
    print("{}{}{}".format(scores[0], Chem.MolToSmiles(population[0]), generation))
    return max(scores)


def axga_evaluation_function(parameterization, weight=None):
    generations, population_size, mating_pool_size = (
        parameterization["generations"],
        parameterization["population_size"],
        parameterization["mating_pool_size"],
    )
    return {"axga": (axga(generations, population_size, mating_pool_size), 0.0)}


exp = SimpleExperiment(
    name="run_axga",
    search_space=axga_search_space,
    evaluation_function=axga_evaluation_function,
    objective_name="axga",
    minimize=False,
)

print("Running Sobol Initialization Batches...")
sobol = modelbridge.get_sobol(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(15):
    print("Running GP+EI optimization batch {i+1}/15...")
    gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
    batch = exp.new_trial(generator_run=gpei.gen(1))
print("Done!")

exp.eval().df
