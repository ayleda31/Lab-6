import time
import numpy as np

try:
    N = int(input("Введите четное количество строк (столбцов) квадратной матрицы больше 3 и меньше 184:"))
    while N < 4 or N > 183 or N % 2 ==1:
        N = int(input("\nВведите четное!!! количество строк (столбцов) квадратной матрицы больше 3!!! и меньше 184!!!:"))
    K = int(input("\nВведите число К:"))
    
    start = time.time()
    A = np.zeros((N, N), dtype=int)
    
    for i in range(N):     
        for j in range(N):
            A[i][j] = np.random.randint(-10, 10)
    print("Матрица A:\n", A)

    F = A.copy()
    n = N // 2
    flag=0
    for i in range(1, n):
        for j in range(i):
            if A[i][j] == A[n-j-1][n-i-1]:
                flag+=1
    
    if flag==1:
        print("|=> Меняем B и D симметрично")
        for i in range(n):       # B и D симметрично
            for j in range(n):
                F[i][n+j] = A[N-i-1][N-j-1]
                F[N-i-1][N-j-1] = A[i][n+j]
    else:
        print("|=> Меняем D и E несимметрично")
        for i in range(n):     # D и E несимметрично
            for j in range(n):
                F[n+i][j] = A[i][j]
                F[i][j] = A[n+i][j]
    print("\nМатрица F:\n", F)
    print("\nОпределитель матрицы А:", round(np.linalg.det(A)), "\nСумма диагональных элементов матрицы F:", np.trace(F))
    if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:
        print("Нельзя вычислить т.к. матрица A или F вырождена")
    elif np.linalg.det(A) > np.trace(F):
        print("\nВычисление выражения: A^-1*A^T-K*F^-1")
        Af = np.dot(np.linalg.inv(A), np.transpose(A)) - (np.linalg.inv(F) * K)  # A^-1*A^T-K*F^-1
    else:
        print("\nВычисление выражения: (A^T+G-F^T)*K")
        Af = (np.transpose(A) + np.tril(A) - np.transpose(F)) * K   # (A^T+G-F^T)*K
    print("\nРезультат:")
    for i in Af:         # Вывод результата
        for j in i:
            print("%5d" % round(j), end=' ')
        print()
    finish = time.time()
    result = finish - start
    print("\nProgram time: "+ str(result) +" seconds.")

except ValueError:
    print("\nЭто не число")
    
#Визуализация данных
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Matplotlib
   #ex. 1
plt.title("Matplotlib ex. 1")
plt.xlabel("index-j")
plt.ylabel("Значения по [i]")
for j in range(N):
    plt.plot([i for i in range(N)], Af[j][::], marker='x')
plt.show()

    #ex. 2
plt.title("Matplotlib ex. 2")
plt.xlabel("index-j")
plt.ylabel("Значения по [i]")
for j in range(N):
    plt.scatter([i for i in range(N)], Af[j][::])
plt.show()

    #ex. 3
figure = plt.figure()
ax = figure.add_subplot()
matrix = ax.matshow(Af, interpolation ='nearest')
figure.colorbar(matrix)
plt.title("Matplotlib ex. 3\n")
plt.xlabel("index-j")
plt.ylabel("index-i")
plt.show()

    #ex. 4
fig, ax = plt.subplots()
c = ax.matshow(Af, cmap ='Greens',vmin = Af.min(),vmax = Af.max(),interpolation ='nearest',origin ='lower')
fig.colorbar(c, ax = ax)
ax.set_title('Matplotlib ex. 4\n')
plt.show()

    #ex. 5
plt.title("Matplotlib ex. 5")
plt.xlabel("index-j")
plt.ylabel("Значения по [i]")
for j in range(N):
    plt.bar([i for i in range(N)], Af[j][::],width = 1)
plt.show()

#Seaborn
   #ex. 1
heatmap = plt.figure(figsize=(N, N)) 
heatmap = sns.heatmap(Af, vmin = Af.min(),vmax = Af.max(), annot=True) 
heatmap.set_title('Seaborn ex. 1', fontdict={'fontsize':12}, pad=12)
plt.show()

    #ex. 2
heatmap = plt.figure(figsize=(N, N)) 
heatmap = sns.heatmap(Af, vmin = Af.min(),vmax = Af.max(),cmap='OrRd_r') 
heatmap.set_title('Seaborn ex. 2', fontdict={'fontsize':12}, pad=12)
plt.show()

    #ex. 3
ax = sns.violinplot(data=Af)
ax.set_title("Seaborn ex. 3")
ax.set_ylabel("Значения по [i]")
ax.set_xlabel("index-j")
plt.show()

