import jiwer
import numpy as np
import pandas as pd
import ptitprince as pt
import matplotlib.pyplot as plt

DS_DIR = '../data/'
metadata = pd.read_csv(DS_DIR + 'ljs-data.csv')
label = 'text1'
models = ['google', 'wit', 'deepgram']

metrics = {}

for m in models:
    metrics[m] = {
        'wer': np.zeros(len(metadata)),
        'mer': np.zeros(len(metadata)),
        'wil': np.zeros(len(metadata)),
        'cer': np.zeros(len(metadata)),
        'errors': np.zeros(len(metadata)).astype(np.bool8),
    }

for index, row in metadata.iterrows():
    print(index, row['wav'])

    for m in models:
        if metadata.iloc[[index]][m].isna().item():
            metrics[m]['errors'][index] = True
        else:
            measures = jiwer.compute_measures(row[label], row[m])
            metrics[m]['wer'][index] = measures['wer']
            metrics[m]['mer'][index] = measures['mer']
            metrics[m]['wil'][index] = measures['wil']
            metrics[m]['cer'][index] = jiwer.cer(row[label], row[m])

for m in models:
    print(m)
    print('wer', metrics[m]['wer'][~metrics[m]['errors']].mean())
    print('mer', metrics[m]['mer'][~metrics[m]['errors']].mean())
    print('wil', metrics[m]['wil'][~metrics[m]['errors']].mean())
    print('cer', metrics[m]['cer'][~metrics[m]['errors']].mean())

models0 = [models[0] for i in range(np.sum(~metrics[models[0]]['errors']))]
models1 = [models[1] for i in range(np.sum(~metrics[models[1]]['errors']))]
models2 = [models[2] for i in range(np.sum(~metrics[models[2]]['errors']))]

d = {
    'model': models0 + models1 + models2,
    'wer': np.concatenate([metrics[models[0]]['wer'][~metrics[models[0]]['errors']],
                           metrics[models[1]]['wer'][~metrics[models[1]]['errors']],
                           metrics[models[2]]['wer'][~metrics[models[2]]['errors']]], 0),
    'mer': np.concatenate([metrics[models[0]]['mer'][~metrics[models[0]]['errors']],
                           metrics[models[1]]['mer'][~metrics[models[1]]['errors']],
                           metrics[models[2]]['mer'][~metrics[models[2]]['errors']]], 0),
    'wil': np.concatenate([metrics[models[0]]['wil'][~metrics[models[0]]['errors']],
                           metrics[models[1]]['wil'][~metrics[models[1]]['errors']],
                           metrics[models[2]]['wil'][~metrics[models[2]]['errors']]], 0),
    'cer': np.concatenate([metrics[models[0]]['cer'][~metrics[models[0]]['errors']],
                           metrics[models[1]]['cer'][~metrics[models[1]]['errors']],
                           metrics[models[2]]['cer'][~metrics[models[2]]['errors']]], 0),
}

df = pd.DataFrame(data=d)
dx = 'model'
ort = 'v'
pal = 'Set2'
sigma = .2

metric_description = {
    'wer': 'word error rating',
    'mer': 'match error rating',
    'wil': 'word information lost',
    'cer': 'character error rating',
}

for metric in ['wer', 'mer', 'wil', 'cer']:
    f, ax = plt.subplots()
    ax = pt.RainCloud(x=dx, y=metric, data=df, palette=pal, bw=sigma,
                      width_viol=.5, ax=ax, orient=ort, move=.3, cut=0)
    plt.ylim([0, 1])

    plt.title(metric_description[metric])
    plt.grid('on', axis='y')
    plt.xlabel('')
    plt.ylabel('')

    plt.show()
