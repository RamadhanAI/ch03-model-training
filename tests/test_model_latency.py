import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.random.rand(1000, 28)
y = np.random.randint(0, 2, 1000)
model = RandomForestClassifier().fit(X, y)
start = time.time()
model.predict(X)
print(f'Latency: {time.time() - start:.2f}s')
