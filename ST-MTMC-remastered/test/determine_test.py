import numpy as np

locations = [[1, 2], [2, 3], [3, 4]]
for i, location1 in enumerate(locations):
    for j, location2 in enumerate(locations):
        if i >= j:
            continue
        d = np.linalg.norm(np.array(location1) - np.array(location2))
        if d <= 2:
            print(str(i) +" " +str(j))

