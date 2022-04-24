import jiwer
import numpy as np
import pandas as pd
import ptitprince as pt
import matplotlib.pyplot as plt

RESULTS_DIR = './results/'
results_google = pd.read_csv(RESULTS_DIR + 'google-results.csv')
results_wit = pd.read_csv(RESULTS_DIR + 'wit-results.csv')
results_deepgram = pd.read_csv(RESULTS_DIR + 'deepgram-results.csv')
results_google_wit = pd.read_csv(RESULTS_DIR + 'google-wit-results.csv')
results_google_deepgram = pd.read_csv(RESULTS_DIR + 'google-deepgram-results.csv')
results_wit_deepgram = pd.read_csv(RESULTS_DIR + 'wit-deepgram-results.csv')
results_google_wit_deepgram = pd.read_csv(RESULTS_DIR + 'google-wit-deepgram-results.csv')
results_google_seq2seq = pd.read_csv(RESULTS_DIR + 'google-seq2seq-results.csv')
results_wit_seq2seq = pd.read_csv(RESULTS_DIR + 'wit-seq2seq-results.csv')
results_deepgram_seq2seq = pd.read_csv(RESULTS_DIR + 'deepgram-seq2seq-results.csv')

models = ['google-raw', 'wit-raw', 'dg-raw', 'google-post-proc', 'wit-post-proc', 'dg-post-proc', 'google-wit', 'google-dg', 'wit-dg', 'google-wit-dg', 'google-seq2seq', 'wit-seq2seq', 'dg-seq2seq']
df = [results_google, results_wit, results_deepgram, results_google, results_wit, results_deepgram, results_google_wit, results_google_deepgram, results_wit_deepgram, results_google_wit_deepgram, results_google_seq2seq, results_wit_seq2seq, results_deepgram_seq2seq]

metrics = {}

for i, m in enumerate(models):
    metrics[m] = {
        'wer': np.zeros(len(df[i])),
        'mer': np.zeros(len(df[i])),
        'wil': np.zeros(len(df[i])),
        'cer': np.zeros(len(df[i])),
    }

    metadata = df[i]

    for index, row in metadata.iterrows():

        input_text = row['input']
        output_text = row['output']
        label_text = row['label']

        if input_text.startswith('translate'):
            input_text = input_text.replace('translate English to English: ', '')
        elif input_text.startswith('merge'):
            input_sentence1 = input_text.replace('merge sentence1: ', '').split(' sentence2: ')[0]
            input_sentence2 = (input_text.replace('merge sentence1: ', '').split(' sentence2: ')[-1]).split(' sentence3: ')[0]
            input_sentence3 = (input_text.replace('merge sentence1: ', '').split(' sentence2: ')[-1]).split(' sentence3: ')[-1]

            if 'dg' in m:
                if m.startswith('dg'):
                    input_text = input_sentence1
                elif m.endswith('dg'):
                    input_text = input_sentence3
                else:
                    input_text = input_sentence2
            else:
                input_text = input_sentence1

        if m.endswith('-raw'):
            output_text = input_text.strip()
        elif m.endswith('-seq2seq'):
            output_text = str(output_text).strip()
        else:
            last_output_word = output_text.strip().replace('.', '').replace(',', '').split(' ')[-1]
            missing_output = input_text.split(last_output_word)[-1]
            output_text = (output_text + missing_output).replace('...', '###').replace(',,', ',').replace('..', '.').replace('###', '...').strip()

        measures = jiwer.compute_measures(label_text, output_text)
        metrics[m]['wer'][index] = measures['wer']
        metrics[m]['mer'][index] = measures['mer']
        metrics[m]['wil'][index] = measures['wil']
        metrics[m]['cer'][index] = jiwer.cer(label_text, output_text)

model_names = None
wer = None
mer = None
wil = None
cer = None
for m in models:
    print(m)
    print('mean')
    print('wer', metrics[m]['wer'].mean())
    print('mer', metrics[m]['mer'].mean())
    print('wil', metrics[m]['wil'].mean())
    print('cer', metrics[m]['cer'].mean())
    print('median')
    print('wer', np.median(metrics[m]['wer']))
    print('mer', np.median(metrics[m]['mer']))
    print('wil', np.median(metrics[m]['wil']))
    print('cer', np.median(metrics[m]['cer']))


# plots = ['google-raw', 'wit-raw', 'dg-raw', 'google-post-proc', 'wit-post-proc', 'dg-post-proc', 'google-wit', 'google-dg', 'wit-dg', 'google-wit-dg']
# plots = ['google-raw', 'google-post-proc', 'google-wit', 'google-dg', 'google-wit-dg']
# plots = ['wit-raw', 'wit-post-proc', 'google-wit', 'wit-dg', 'google-wit-dg']
# plots = ['dg-raw', 'dg-post-proc', 'google-dg', 'wit-dg', 'google-wit-dg']
plots = []

for m in plots:
    if model_names is None:
        model_names = np.repeat(m, len(metrics[m]['wer']))
        wer = metrics[m]['wer']
        mer = metrics[m]['mer']
        wil = metrics[m]['wil']
        cer = metrics[m]['cer']
    else:
        model_names = np.concatenate([model_names, np.repeat(m, len(metrics[m]['wer']))], axis=0)
        wer = np.concatenate([wer, metrics[m]['wer']], axis=0)
        mer = np.concatenate([mer, metrics[m]['mer']], axis=0)
        wil = np.concatenate([wil, metrics[m]['wil']], axis=0)
        cer = np.concatenate([cer, metrics[m]['cer']], axis=0)

if len(plots) > 0:
    d = {
        'model': model_names,
        'wer': wer,
        'mer': mer,
        'wil': wil,
        'cer': cer,
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
