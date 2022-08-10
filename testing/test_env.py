from environment import ScreenEnv
import time

env = ScreenEnv()

count = 0
while True:
    t = time.time()
    env.get_state()
    print(1/(time.time()-t))
    count += 1
    if count > 100000:
        exit()