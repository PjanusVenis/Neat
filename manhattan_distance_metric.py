from typing import List, Tuple, Dict


class ManhattanDistanceMetric:
    def __init__(self, match_coeff: float = 1.0, mismatch_coeff: float = 1.0, mismatch_constant: float = 0.0):
        self.match_coeff = match_coeff
        self.mismatch_coeff = mismatch_coeff
        self.mismatch_constant = mismatch_constant

    def measure_distance(self, pos1: List[Tuple[int, float]], pos2: List[Tuple[int, float]]) -> float:
        pos1_length = float(len(pos1))
        pos2_length = float(len(pos2))

        if pos1_length == pos2_length == 0:
            return 0.0

        distance = 0

        if pos1_length == 0:
            for (_, val) in pos2:
                distance += abs(val)
            return self.mismatch_coeff * (pos2_length + distance)

        if pos2_length == 0:
            for (_, val) in pos1:
                distance += abs(val)
            return self.mismatch_coeff * (pos1_length + distance)

        pos1_idx = 0
        pos2_idx = 0

        while True:
            pos1_inno, pos1_weight = pos1[pos1_idx]
            pos2_inno, pos2_weight = pos2[pos2_idx]

            if pos1_inno < pos2_inno:
                distance += self.mismatch_constant + (abs(pos1_weight) * self.mismatch_coeff)
                pos1_idx += 1
            elif pos1_inno == pos2_inno:
                distance += abs(pos1_weight - pos2_weight) * self.match_coeff
                pos1_idx += 1
                pos2_idx += 1
            else:
                distance += self.mismatch_constant + (abs(pos2_weight) * self.mismatch_coeff)
                pos2_idx += 1

            if pos1_idx == pos1_length:
                for i in range(pos2_idx, int(pos2_length)):
                    _, pos2_weight = pos2[i]
                    distance += self.mismatch_constant + (abs(pos2_weight) * self.mismatch_coeff)
                return distance

            if pos2_idx == pos2_length:
                for i in range(pos1_idx, int(pos1_length)):
                    _, pos1_weight = pos1[i]
                    distance += self.mismatch_constant + (abs(pos1_weight) * self.mismatch_coeff)
                return distance

    def calculate_centroid(self, pos_list: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        centroid_pos_helper: List[(float, int, int)] = []
        centroid_pos_dict: Dict[int: int] = {}

        for genome_pos in pos_list:
            for pos_id, pos_weight in genome_pos:
                if pos_id in centroid_pos_dict:
                    helper_weight, helper_count, helper_inno = centroid_pos_helper[centroid_pos_dict[pos_id]]
                    centroid_pos_helper[centroid_pos_dict[pos_id]] = (helper_weight + pos_weight, helper_count + 1, helper_inno)
                else:
                    idx = len(centroid_pos_helper)
                    centroid_pos_helper.append((pos_weight, 1, pos_id))
                    centroid_pos_dict[pos_id] = idx

        return [(c, a / b) for a, b, c in centroid_pos_helper]


