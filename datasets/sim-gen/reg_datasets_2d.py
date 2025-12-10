
import andi
import numpy as np
import pandas as pd
import argparse
import gc

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--N', type=int)
arg('--l', type=int)
args = parser.parse_args()

N = args.N
l = args.l
filename = './origin_data/data-2d-{}.csv'.format(l)
output = './pp_data_2d/data-2d-{}-pp.csv'.format(l)

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=N, max_T=l+1, min_T=l, tasks=1, dimensions=2)

with open(filename, 'w') as f:
    f.write('pos;label\n')
    for i in range(len(X1[1])):
        f.write(','.join([str(j) for j in X1[1][i]]))
        f.write(';'+str(Y1[1][i])+'\n')
    f.close()

del X1, Y1
gc.collect()

data = pd.read_csv(filename, sep=';')
data['length'] = data['pos'].apply(lambda x: round(len(x.split(','))/2))

data['pos_x'] = data.apply(lambda x: ','.join(x['pos'].split(',')[:x['length']]), axis=1)
data['pos_y'] = data.apply(lambda x: ','.join(x['pos'].split(',')[x['length']:]), axis=1)

del data['pos']

def normalizex(df):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    data = np.array([[float(i)] for i in df['pos_x'].split(',')])
    scaler = StandardScaler()
    scaler.fit(data)
    data2 = scaler.transform(data)
    data2 = data2.reshape(-1)
    return ','.join([str(round(i,6)) for i in data2])

def normalizey(df):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    data = np.array([[float(i)] for i in df['pos_y'].split(',')])
    scaler = StandardScaler()
    scaler.fit(data)
    data2 = scaler.transform(data)
    data2 = data2.reshape(-1)
    return ','.join([str(round(i,6)) for i in data2])

data['pos_x'] = data.apply(normalizex, axis=1)
data['pos_y'] = data.apply(normalizey, axis=1)

data[['pos_x','pos_y','length','label']].to_csv(output, index=False, sep=';')
