# -!- coding: utf-8 -!-
import json
import os
import requests
from nltk.tokenize import sent_tokenize
import openai
import pandas as pd

openai.api_key = '' #FILL IN YOUR OWN HERE


prompt_abstract="I want you to act as a Caption Evaluator. I will provide you with an audio caption generated by an AI agent. The agent was asked to generate a dense and detailed caption of the audio. To evaluate the caption, I will provide you with 2 different types of information about the 10-second audio clip:\n\n1. A list where each comma-separated element indicates the individual events occurring in the audio at various time segments. For example, '(Speech-0.0-0.64)' would mean human speech between 0.0 second to 0.64 second.\n2. A scene caption of the audio describing in a brief and abstract manner the scene in which the audio takes place. Using these two pieces of information, assign a score of 1-10 to the caption, where 1 is the lowest score and 10 is the highest score. Your evaluation should be based on the detailedness, correctness, and bluntness of the caption. Return a JSON with a single key 'score', where the value of the key is the score. Here are the details:"

def prompt_gpt(prompt_input, json_output = True):

    response = openai.ChatCompletion.create(model="gpt-4o",
                                                messages=[{"role": "user", "content": prompt_input}],
                                                temperature=0.5,
                                                max_tokens=4096,
                                                response_format={ "type": "json_object" }
                                                )

    return response


def get_llm_summary():

    in_file = "./stage_5_captions.json"
    in_file = open(in_file, 'r')
    in_file = json.load(in_file)

    # acd = pd.read_csv(in_file)

    for index, row in enumerate(in_file):
        
        audio_id = row["audio_id"]
        caption = row["prediction"]
        timestamp_events = row["timestamp_events"]
        scene_caption = row['scene_caption']

        try:
            x = prompt_abstract + "\nInput list of audio events: " + timestamp_events + "\nScene Caption: " + scene_caption + "\nCaption by agent: " + caption

            response = prompt_gpt(x)
            pred = response['choices'][0]['message']['content'].replace("\n","")

            print(pred)

            data = {"id": audio_id,
            "caption": caption,
            "timestamps": timestamp_events}
            pred = eval(pred)
            data.update(pred)
            with open("caption_score.json", "a") as g:
                g.write(json.dumps(data) + '\n')
        except Exception as e:
            print(e)
            continue


def process_input(lines):

    all_sentences = sent_tokenize(lines)
    segments_new = {}

    for i,item in enumerate(all_sentences):
        segments_new[i] = item

    return segments_new

def post_process_input(src,response_segments,response_key):

    all_sentences = sent_tokenize(src)

    for key,value in response_key.items():
        sent = all_sentences[int(key)]
        for phrase in value: 
            sent = all_sentences[int(key)]
            start = sent.lower().find(phrase.lower())
            if start == "-1":
                continue
            end = start + len(phrase)
            sent = sent[:start-1] + " **" + phrase + "** " + sent[end+1:]

        all_sentences[int(key)] = sent
    
    
    article_wo = " ".join(all_sentences)

    all_keys = []
    article = ""
    for key,value in response_segments.items():
        all_keys.extend(value)

    if len(all_keys) != len(all_sentences):
        print("difference")

    for key,value in response_segments.items():
        article += "[ "
        for sent in response_segments[key]:
            article += all_sentences[int(sent)]
        article += " ]"

    return article_wo, article



if __name__ == '__main__':

    get_llm_summary()
