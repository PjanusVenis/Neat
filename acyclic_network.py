from typing import List, Tuple


class AcyclicNetwork:
    def __init__(self, activation_fns: [], connections: List[Tuple[int, int, float]], layer_info: List[Tuple[int, int]],
                 output_node_idx_list: List[int], node_count: int, input_count: int, output_count: int):
        self.activation_fns = activation_fns
        self.connections = connections
        self.layer_info = layer_info
        self.output_node_idx_list = output_node_idx_list
        self.node_count = node_count
        self.input_count = input_count
        self.output_count = output_count

        self.activation_list: List[float] = [0.0 for _ in range(self.node_count)]
        self.activation_list[0] = 1.0

        self.input_offset = 1
        self.output_signal_mapping: List[int] = self.output_node_idx_list

        self.input_bias_count = input_count + 1

    def activate(self):
        for i in range(self.input_bias_count, self.node_count):
            self.activation_list[i] = 0.0

        con_idx = 0
        node_idx = self.input_bias_count

        for i in range(1, len(self.layer_info)):
            _, end_connection_idx = self.layer_info[i - 1]

            for _ in range(con_idx, end_connection_idx):
                from_id, to_id, weight = self.connections[con_idx]

                self.activation_list[to_id] += self.activation_list[from_id] * weight
                con_idx += 1

            end_node_idx, _ = self.layer_info[i]

            for _ in range(node_idx, end_node_idx):
                self.activation_list[node_idx] = self.activation_fns[node_idx](self.activation_list[node_idx])

    def set_input(self, inputs: List[float]):
        if len(inputs) != self.input_count:
            return

        self.activation_list[self.input_offset:self.input_offset + self.input_count] = inputs

    def get_output(self) -> List[float]:
        return [self.activation_list[a] for a in self.output_signal_mapping]


