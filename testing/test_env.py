from environment import ScreenEnv
import time

env = ScreenEnv()

while True:
    t = time.time()
    env.get_state()
    print(1/(time.time()-t))