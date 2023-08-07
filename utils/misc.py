

def is_power_of_two(x):
    return (x & (x-1) == 0) and x > 0
