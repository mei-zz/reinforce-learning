import pandas as pd
import numpy as np

"""
Pandas模块的数据结构主要有两：1、Series;2、DataFrame
series是一个一维数组，是基于NumPy的ndarray结构。
Pandas会默然用0到n-1来作为series的index，但也可以自己指定index(可以把index理解为dict里面的key)。
"""
s = pd.Series(np.nan,index=[49,48,47,46,45,1,2,3,4,5])
# print(s)
print(s.iloc[:3])   #这儿iloc代表的是索引值
print("******************")
print(s.loc[:3])    #这是的loc代表是label值，所以直接会取到label为3
print("******************")
# print(s.ix[:3])   #ix已经无法使用！！！！！！！