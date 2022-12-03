import pandas as pd
import numpy as np

se1 = pd.Series([1,7,3,9],index=['b','c','a','d'])
print(se1)
print("********************")
se2 = se1.reindex(['a','b','c','d','e','f'])
print(se2)
