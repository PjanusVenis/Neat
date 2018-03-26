class SpecieStats:
    def __init__(self):
        self.mean_fitness = 0
        self.mean_novelty = 0
        self.target_size_real = 0
        self.target_size_int = 0
        self.elite_size_int = 0
        self.offspring_count = 0
        self.offspring_asexual_count = 0
        self.offspring_sexual_count = 0
        self.selection_size_int = 0