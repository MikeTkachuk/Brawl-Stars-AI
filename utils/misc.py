

def is_power_of_two(x):
    return (x & (x-1) == 0) and x > 0


def create_token(inp, n_binary=4, anchors=(4,4)):
    assert len(inp) == n_binary + len(anchors)
    bin_str = ''
    for i in range(n_binary):
        bin_str += str(inp[i])

    full_size = len(bin(anchors[0]-1))-2
    anch_bin = bin(inp[-2])[2:]
    bin_str += '0' * (full_size - len(anch_bin)) + anch_bin

    full_size = len(bin(anchors[1]-1))-2
    anch_bin = bin(inp[-1])[2:]
    bin_str += '0' * (full_size - len(anch_bin)) + anch_bin

    assert len(bin_str) == len(bin(2**n_binary*anchors[0]*anchors[1])) - 3, "Incorrect token range"
    return int(bin_str, 2)
