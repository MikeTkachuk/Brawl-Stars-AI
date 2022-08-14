from environment import ScreenParser
import time

env = ScreenParser()

count = 0
while True:
    t = time.time()
    env.get_state()
    print(1/(time.time()-t))
    count += 1
    if count > 100000:
        exit()
