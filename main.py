import functions.circle as circ
import os
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    r = 5
    area = circ.area(radius=r)
    print('The area of a circle with radius %f is %f'%(r,area))

    abs_path = os.path.abspath(__file__ + "/../_data/data.csv")
    data = pd.read_csv(abs_path)
    data['area'] = data.apply(lambda x: circ.area(x['radius']), axis=1)
    print()