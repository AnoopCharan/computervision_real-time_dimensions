import numpy as np

mypoints = np.random.randint(1,10,(4,1,2))

print(mypoints)
print(mypoints[1])

mypoints_new = np.zeros_like(mypoints)

mypoints = mypoints.reshape((4,2))

add = mypoints.sum(1)

min = np.argmin(add)

diff= np.diff(mypoints, axis=1)


# print(add)
# print(min)
# print(diff)