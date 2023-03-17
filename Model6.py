'''
Лабораторна №6 моделювання Нау 4 курс
'''


import pandas as pd
import numpy as np
import itertools
import math as m
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import f
from statistics import stdev
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#..Вхідний ЧР
y=np.array([83, 80, 79, 75, 70, 74, 72, 70, 65, 66, 69, 59, 60, 59, 56, 54, 53, 50])
#...вносимо вхідні дані
p=2
alfa=0.1
#...Розділяємо наш чр на 2 рівні
n1=y.size//2
y1 = y[:n1]
y2 = y[n1:]

#...рахуємо середні значення рівнів
S1=np.mean(y1)
S2=np.mean(y2)
#...рахуємо дисперсії рівнів
D1=stdev(y1)**2
np.var(y1)
np.var(y2)
D2=stdev(y2)**2


#...визначаємо значення критерію Фішера 
if D1>=D2:
    F=D1/D2
else:
    F=D2/D1
print(f'Критерій Фішера F= {F}\n')
print(f'Табличне значення Ftab = {f.ppf(1-alfa, n1-1,n1-1)}\n')

if F<(f.ppf(1-alfa, n1-1,n1-1)):
    print('Гіпотеза про відсутність тренду  приймається, у базовому ЧР\n')
else:
    print('Гіпотеза про відсутність тренду не приймається, у базовому ЧР\n')   
    



Skv=(((n1-1)*D1+(n1-1)*D2)/(n1+n1-2))**0.5
print(f'Cередньоквадратичне відхилення різності середніх Skv= {Skv}\n')


tr=(abs(S1-S2))/(Skv*(1/n1 + 1/n1)**0.5)
tk=(t.ppf(1-alfa, y.size-p))

if tk>tr:
    print(f'Тренд відсутній у базовому ЧР {tk} > {tr}\n')
else:
    print(f'Тренд присутній у базовому ЧР {tk} < {tr}\n')

# №1
def f_seria(y):
    for i in y:
        if i>np.median(y):
            yield 1
        else:
            yield 0

Y=[n for n in f_seria(y)]

# №2 Y=[1 if i>np.median(y) else 0 for i in y]
       
#...       
a=[(len(list(group))) for i, group in itertools.groupby(Y)]
#...
A=pd.Series(a,[('seria'+str(i)) for i in range(1,len(a)+1)])

print(pd.DataFrame(A).T) # створюємо DataFrame та транспонуємо його
print("\n")
#...
a=np.array(a)
print(f'максимальная довжина серії = {np.max(a)}\n')
print(f'кількість серій = {np.size(a)}\n')

#...
if np.size(a)>(n1*2+2-(3.92*n1)**0.5) and np.max(a)<=abs(3.3*m.log10(n1*2+1)):
    print('Тренд в чс відсутній')
else:
    print('Тренд присутній')
    
#...
yz=[y[0]]    
for i in range(1,len(y)):
    yz.append(alfa*y[i]+(1-alfa)*y[i-1])
else:
     yz=np.array(yz)
     print(yz)


#//створюємо DataFrame з двома стовпцями та будуємо графік
y_table=pd.DataFrame({'y_базове':y,'y_зглажене':yz})
plt.figure(figsize=(12,9))
y_table.plot(y=['y_зглажене','y_базове'])
plt.show()
#...
x = np.arange(0, yz.size)

poly = PolynomialFeatures(degree=2) 

poly.fit_transform(x.reshape(-1,1))

#//Створення об'єкту poly_reg_model класу LinearRegression.
poly_reg_model = LinearRegression()

#//Виклик методу fit для об'єкту poly_reg_model з параметром poly.fit_transform(x.reshape(-1,1)) та yz для побудови моделі регресії.
poly_reg_model.fit(poly.fit_transform(x.reshape(-1,1)), yz)

#//Виклик методу predict для об'єкту poly_reg_model з параметром poly.fit_transform(x.reshape(-1,1)) для отримання передбачених значень.
y_predicted = poly_reg_model.predict(poly.fit_transform(x.reshape(-1,1)))

#//Додавання колонки 'y_polinom' з передбаченими значеннями y_predicted до таблиці y_table.
y_table['y_polinom']=y_predicted

#//Побудова графіку за допомогою методу plot для колонок 'y_зглажене', 'y_базове' та 'y_polinom' з таблиці y_table. 
y_table.plot(y=['y_зглажене','y_базове','y_polinom'],title='Графік моделей',figsize=(9, 6),color=['r','y','b'])

#//Обчислення середньоквадратичного відхилення Sy
Sy=((sum((yz-y_predicted)**2))/(y.size-p-1))**0.5
print(Sy)

#//Створення масиву xp з одним значенням yz.size + 9.
xp=np.array(yz.size + 9)

#//Виклик методу predict для об'єкту poly_reg_model з параметром xp.reshape(-1,1) для отримання передбаченого значення для нової точки.
y_point = poly_reg_model.predict(poly.fit_transform(xp.reshape(-1,1)))

#//
plt.Figure()
plt.figure(figsize=(10,6))
plt.plot(x, y_table['y_polinom'])
plt.plot(x, y_table['y_зглажене'])
plt.scatter(xp, y_point)
Del1=y_point + tk*Sy
Del2=y_point - tk*Sy


plt.scatter(xp, Del1,
            marker = '^',)
plt.scatter(xp, Del2,
            marker = 'v',)







