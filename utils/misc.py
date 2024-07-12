import numpy as np


class ActionTokenizer:
    def __init__(self, n_binary_actions=3, move_shot_anchors=(4, 4)):
        self.n_binary_actions = n_binary_actions
        self.move_shot_anchors = move_shot_anchors if hasattr(move_shot_anchors, "__len__") else (
                                                                                                 move_shot_anchors,) * 2

        self.n_actions = 2 ** self.n_binary_actions * (1 + move_shot_anchors[0]) * move_shot_anchors[1]
        self.bit_space = (len(bin(self.move_shot_anchors[0])) - 2,
                          len(bin(self.move_shot_anchors[1] - 1)) - 2)  # anchors only

        self._action_binary_maps = None

    def _action_map_binary(self, token, input_binary=False):
        """
        Converts between binary and consecutive formats. Binary is ready to be split into bins and parsed.
        Consecutive is basically an id of an action.
        :param token: int, action token
        :param input_binary: if True, assumes token to be in binary format
        :return: token in the opposite format
        """
        if self._action_binary_maps is None:
            b_tokens = []
            total_bits = self.n_binary_actions + sum(self.bit_space)
            for i in range(2 ** total_bits):  # filter valid tokens
                bin_i = bin(i)[2:]
                bin_i = '0' * (total_bits - len(bin_i)) + bin_i
                bin_move_anchor = bin_i[self.n_binary_actions:][:self.bit_space[0]]
                if int(bin_move_anchor, 2) > self.move_shot_anchors[0]:
                    continue
                bin_shot_anchor = bin_i[-self.bit_space[1]:]
                if int(bin_shot_anchor, 2) >= self.move_shot_anchors[1]:
                    continue
                b_tokens.append(i)
            c_tokens = range(len(b_tokens))
            self._action_binary_maps = (
                dict(zip(c_tokens, b_tokens)),
                dict(zip(b_tokens, c_tokens))
            )

        if input_binary:
            return self._action_binary_maps[1][token]
        else:
            return self._action_binary_maps[0][token]

    def create_action_token(self, make_move, make_shot, super_ability, use_gadget,
                            move_anchor, shot_anchor, ):
        assert 0 <= move_anchor < self.move_shot_anchors[0]
        assert 0 <= shot_anchor < self.move_shot_anchors[1]
        move_anchor = move_anchor + 1 if make_move else 0

        bin_str = ''
        for i in [make_shot, super_ability, use_gadget]:
            bin_str += str(int(i))

        anch_bin = bin(move_anchor)[2:]
        bin_str += '0' * (self.bit_space[0] - len(anch_bin)) + anch_bin

        anch_bin = bin(shot_anchor)[2:]
        bin_str += '0' * (self.bit_space[1] - len(anch_bin)) + anch_bin

        assert len(bin_str) == self.n_binary_actions + sum(self.bit_space), "Incorrect token range"
        b_token = int(bin_str, 2)
        return self._action_map_binary(b_token, input_binary=True)

    def split_into_bins(self, token: int, input_binary=False, decode_move_anchor=True):
        if not input_binary:
            token = self._action_map_binary(token, input_binary=False)
        out = []
        bin_t = bin(token)[2:]
        total_bits = self.n_binary_actions + sum(self.bit_space)
        bin_t = '0' * (total_bits - len(bin_t)) + bin_t  # pad with 0

        for i in range(self.n_binary_actions):
            out.append(int(bin_t[i]))
        move_anchor = int(bin_t[self.n_binary_actions:][:self.bit_space[0]], 2)
        shot_anchor = int(bin_t[-self.bit_space[1]:], 2)
        out += [move_anchor, shot_anchor]
        if decode_move_anchor:  # add legacy make_move
            out = [int(move_anchor > 0)] + out
            if move_anchor > 0:
                out[-2] -= 1  # make move anchors in 0-n
        return out

    def parse_action_token(self, action):
        """
        Parse 1 consecutive action token and 3 continuous action values
        :param action: array-like of action values
        :return: dict of parsed actions
        """
        assert len(action) == 4, "Wrong action format"  # 1 token + 3 continuous [0,1]: (move, shot, strength)
        assert self.n_binary_actions == 3, "Only 3 binary actions supported. See docs"
        assert 0 <= action[0] < self.n_actions, "Action token out of bound"
        bin_action = self._action_map_binary(action[0])
        bins = bin(int(bin_action))[2:]
        total_bits = self.n_binary_actions + sum(self.bit_space)
        bins = '0' * (total_bits - len(bins)) + bins  # pad with 0

        make_shot, super_ability, use_gadget = bins[:self.n_binary_actions]

        move_anchor = int(bins[self.n_binary_actions:][:self.bit_space[0]], 2)
        shot_anchor = int(bins[-self.bit_space[1]:], 2)

        def _get_anchor_dir(anchor_num, total, shift=0.0):
            assert anchor_num < total
            angle_shift = 1 / total * 2 * np.pi * (shift - 0.5)  # assumes shift is in [0, 1]
            angle = anchor_num / total * 2 * np.pi + angle_shift
            anchor = np.array([np.cos(angle), np.sin(angle)])
            return anchor

        parsed_action = {
            'direction': _get_anchor_dir(max(move_anchor - 1, 0), self.move_shot_anchors[0], action[1]),  # 0 is no_move
            'make_move': int(move_anchor > 0),
            'make_shot': int(make_shot),
            'shoot_direction': _get_anchor_dir(shot_anchor, self.move_shot_anchors[1], action[2]),
            'shoot_strength': action[3],
            'super_ability': int(super_ability),
            'use_gadget': int(use_gadget),
        }
        return parsed_action


if __name__ == "__main__":
    from environment import GymEnv, make_env
    e = make_env(move_shot_anchors=4)
    a = ActionTokenizer()
    print(e.parse_action_token([159, 0.5, 0.5, 0]))
    for i in range(a.n_actions):
        print(a.create_action_token(*a.split_into_bins(i)) == i)
        print(a.split_into_bins(i))
        print(e.parse_action_token([i, 0.5, 0.5, 0]))