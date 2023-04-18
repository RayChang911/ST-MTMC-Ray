from fastdtw import fastdtw
import matplotlib.pyplot as plt

ts1 = [[1, 1], [2, 2]]
ts2 = [[1, 2], [2, 9]]
ts3 = [[1, 7], [2, 10], [3, 15]]
distance1, path1 = fastdtw(ts1, ts2)
print('ts1 vs. ts2:',distance1)

distance2, path2 = fastdtw(ts1, ts3)
print('ts1 vs. ts3:',distance2)

distance3, path3 = fastdtw(ts2, ts3)
print('ts2 vs. ts3:',distance3)

plt.plot(ts1, label = "ts1")
plt.plot(ts2, label = "ts2")
plt.plot(ts3, label = "ts3")
plt.show()