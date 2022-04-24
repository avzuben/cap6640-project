import asyncio
import pandas as pd
import traceback

from recognize import recognize_google, recognize_wit, recognize_deepgram

DS_DIR = './LJSpeech-1.1/'
WAV_DIR = DS_DIR + 'wavs/'
WAV_SUFFIX = '.wav'


async def main():
    column_names = ['wav', 'text1', 'text2', 'google', 'wit', 'deepgram']
    metadata = pd.read_csv(DS_DIR + 'metadata.csv', sep='|', names=column_names, dtype='object')

    for index, row in metadata.iterrows():
        print(index, row['wav'])
        # GOOGLE
        try:
            rec = await recognize_google(WAV_DIR + row['wav'] + WAV_SUFFIX)
        except:
            print(traceback.format_exc())
            rec = ""
        metadata.at[index, 'google'] = rec
        # print('google', rec)
        # WIT
        try:
            rec = await recognize_wit(WAV_DIR + row['wav'] + WAV_SUFFIX)
        except:
            print(traceback.format_exc())
            rec = ""
        metadata.at[index, 'wit'] = rec
        # print('wit', rec)
        # DEEPGRAM
        try:
            rec = await recognize_deepgram(WAV_DIR + row['wav'] + WAV_SUFFIX)
        except:
            print(traceback.format_exc())
            rec = ""
        metadata.at[index, 'deepgram'] = rec
        # print('deepgram', rec)
        # break

    metadata.to_csv(DS_DIR + 'new-metadata.csv')


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
