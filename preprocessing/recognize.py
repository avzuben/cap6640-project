import json
import aiohttp
import speech_recognition as sr

from deepgram import Deepgram
from urllib.parse import urlencode

r = sr.Recognizer()
WIT_KEY = ''  # WIT app key
DEEPGRAM_SECRET = ''  # Deepgram secret key

dg_client = Deepgram(DEEPGRAM_SECRET)


def wav_filename_2_audio_data(filename):
    with sr.AudioFile(filename) as source:
        audio_listened = r.listen(source)
    return audio_listened


async def recognize_wit(filename, show_all=False):
    audio_data = wav_filename_2_audio_data(filename)

    key = WIT_KEY

    wav_data = audio_data.get_wav_data(
        convert_rate=16000,  # audio samples should be 16 kHz
        convert_width=2  # audio samples should be 16-bit
    )

    url = "https://api.wit.ai/speech?v=20220306"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=wav_data,
                      headers={"Authorization": "Bearer {}".format(key), "Content-Type": "audio/wav"}) as response:

            response_text = await response.text('utf-8')
            # print(response_text)
            result = json.loads(response_text[response_text[:response_text.find('"entities"')].rfind('{'):response_text.rfind('}')+1])

    # return results
    if show_all: return result
    if "text" not in result or result["text"] is None: raise Exception(response_text)
    return result["text"]


async def recognize_google(filename, key=None, pfilter=0, show_all=False):
    audio_data = wav_filename_2_audio_data(filename)

    language = "en-US"

    flac_data = audio_data.get_flac_data(
        # convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
        convert_rate=8000,  # audio samples must be at least 8 kHz
        convert_width=2  # audio samples must be 16-bit
    )
    if key is None: key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
    url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
        "client": "chromium",
        "lang": language,
        "key": key,
        "pFilter": pfilter
    }))

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=flac_data,
                      headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)}) as response:

            response_text = await response.text('utf-8')
            # print(response_text)

    # ignore any blank blocks
    actual_result = []
    for line in response_text.split("\n"):
        if not line: continue
        result = json.loads(line)["result"]
        if len(result) != 0:
            actual_result = result[0]
            break

    # return results
    if show_all:
        return actual_result
    if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0:
        raise Exception(actual_result)

    if "confidence" in actual_result["alternative"]:
        # return alternative with highest confidence score
        best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
    else:
        # when there is no confidence available, we arbitrarily choose the first hypothesis.
        best_hypothesis = actual_result["alternative"][0]
    if "transcript" not in best_hypothesis:
        raise Exception(best_hypothesis)
    return best_hypothesis["transcript"]


async def recognize_deepgram(filename, show_all=False):
    audio_data = wav_filename_2_audio_data(filename)

    lang = 'en-US'

    wav_data = audio_data.get_wav_data(
        convert_rate=16000,  # audio samples should be 16 kHz
        convert_width=2  # audio samples should be 16-bit
    )
    source = {'buffer': wav_data, 'mimetype': 'audio/wav'}
    result = await dg_client.transcription.prerecorded(source, {'language': lang, 'punctuate': True})

    if show_all: return result
    if "results" in result:
        if "channels" in result["results"]:
            if len(result["results"]["channels"]) > 0:
                if "alternatives" in result["results"]["channels"][0]:
                    if len(result["results"]["channels"][0]["alternatives"]) > 0:
                        if "transcript" in result["results"]["channels"][0]["alternatives"][0]:
                            return result["results"]["channels"][0]["alternatives"][0]["transcript"]
                        else:
                            raise Exception(result)
                    else:
                        raise Exception(result)
                else:
                    raise Exception(result)
            else:
                raise Exception(result)
        else:
            raise Exception(result)
    else:
        raise Exception(result)
