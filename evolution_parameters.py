class EvolutionParameters:
    def __init__(self, specie_count: int, elitism_proportion: float, selection_proportion: float,
                 offspring_asexual_proportion: float, offspring_sexual_proportion: float,
                 offspring_interspecies_proportion: float):
        self.specie_count = specie_count
        self.elitism_proportion = elitism_proportion
        self.selection_proportion = selection_proportion
        self.offspring_asexual_proportion = offspring_asexual_proportion
        self.offspring_sexual_proportion = offspring_sexual_proportion
        self.offspring_interspecies_proportion = offspring_interspecies_proportion


default_evolution_parameters = EvolutionParameters(5, 0.2, 0.2, 0.5, 0.49, 0.01)