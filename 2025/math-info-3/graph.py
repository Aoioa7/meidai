import numpy as np
import matplotlib.pyplot as plt

h = 0.1
x = []
u_num = []
with open('data.dat') as f:
	for line in f:
		line = line.strip()
		if not line:
			continue
		parts = line.replace('i=', '').replace('u=', '').split()
		idx = int(parts[0])
		u = float(parts[1])
		x.append(idx * h)
		u_num.append(u)
		
x = np.array(x)
u_num = np.array(u_num)

# 厳密解の計算
B = -np.cos(1.0) / np.sin(1.0)
u_exact = np.cos(x) + B * np.sin(x)

# 数値解と厳密解を重ね描き
plt.figure()
plt.plot(x, u_num, 'o-', label='Numerical')
plt.plot(x, u_exact, '-', label='Exact')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Numerical Solution vs Exact Solution')
plt.grid(True)
plt.legend()
plt.show()
