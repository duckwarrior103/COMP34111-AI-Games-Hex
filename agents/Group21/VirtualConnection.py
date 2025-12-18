class VirtualConnection():
    def __init__(self, board_size):
        self.N = board_size;
        self.bridge_offsets = [
            (-1, 2), (1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1)
        ]

        self.carrier_offsets = {
            (-1, 2): [(0, 1), (-1, 1)],
            (1, 1):  [(0, 1), (1, 0)],
            (2, -1): [(1, 0), (1, -1)],
            (1, -2): [(0, -1), (1, -1)],
            (-1, -1):[(0, -1), (-1, 0)],
            (-2, 1): [(-1, 0), (-1, 1)]
        }

    def get_index(self, r, c):
        return r * self.N + c

    def find_all_bridges(self, board, player_colour):
        """Returns a list of active bridges for a player."""
        active_bridges = []
        player_stones = board.get_player_cells(player_colour)
        
        for stone_row, stone_col in player_stones:
            # Check each possible bridge pattern from this stone
            for bridge_row_offset, bridge_col_offset in self.bridge_offsets:
                # Calculate where the other endpoint would be
                neighbor_row = stone_row + bridge_row_offset
                neighbor_col = stone_col + bridge_col_offset
                
                # Check if neighbor is on the board
                if 0 <= neighbor_row < self.N and 0 <= neighbor_col < self.N:
                    # Check if neighbor stone is same colour, forms an endpoint
                    if board.get_colour(neighbor_row, neighbor_col) == player_colour:
                        # Get the two carrier cells, bridge holes
                        carrier_offset_1, carrier_offset_2 = self.carrier_offsets[(bridge_row_offset, bridge_col_offset)]
                        
                        carrier_row_1 = stone_row + carrier_offset_1[0]
                        carrier_col_1 = stone_col + carrier_offset_1[1]
                        carrier_row_2 = stone_row + carrier_offset_2[0]
                        carrier_col_2 = stone_col + carrier_offset_2[1]
                        
                        # Check if both carrier cells are empty, bridge is active
                        if board.is_empty(carrier_row_1, carrier_col_1) and \
                        board.is_empty(carrier_row_2, carrier_col_2):
                            active_bridges.append({
                                'endpoints': (
                                    self.get_index(stone_row, stone_col),
                                    self.get_index(neighbor_row, neighbor_col)
                                ),
                                'carrier': [
                                    self.get_index(carrier_row_1, carrier_col_1),
                                    self.get_index(carrier_row_2, carrier_col_2)
                                ]
                            })
        
        return active_bridges