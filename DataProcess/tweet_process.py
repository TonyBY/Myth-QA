try:
    from Helper.utils import read_jsonl, save_jsonl, get_raw, logger, init_logger
    from Helper.TextNormalization import normalization_pipeline
except:
    import sys
    from pprint import pprint
    print("sys.path:")
    pprint(sys.path)
    print("\nBefore running the code, make sure the project dir is in sys.path.")
    print("Otherwise, run 'export PYTHONPATH=$<your project dir>' in your ternimal first.\n")
    exit()

import os
import re
import glob
import json
import argparse

def define_dsearch_args(parser):
    parser.add_argument('--keywords', 
                        type=str, 
                        metavar="A list of keywords used to select files to process, e.g., 'dogs,spread'.", 
                        nargs='+',
                        default=[],
                        help="A list of keywords used to select files to process, e.g., 'dogs,spread'.")
    parser.add_argument('--exclude-keywords', type=str, metavar="A list of keywords used to select files not to process, e.g., 'COVID-19,Iraq War'.",
                        nargs='+',
                        default=[],
                        help="A list of keywords used to select files not to process, e.g., 'COVID-19,Iraq War'.")
    
    parser.add_argument('--input-path', 
                        type=str, 
                        default='../data/raw/tweets/', 
                        help='Directory saving raw tweets.')

    parser.add_argument('--output_path', 
                        type=str, 
                        default='../data/processed_tweets/', 
                        help='Dirctory saving json files of processed tweets.')

def get_files_in_dir(directory='../TwitterAPI/data/tweets/'):
    return [f for f in glob.glob(directory + "*.json")]

def read_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def save_json(json_obj, json_path='./data/processed_tweets.json'):
    jsonFile =  open(json_path, 'w')
    jsonFile.write(json.dumps(json_obj))
    jsonFile.close()

def remove_url(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_tags(text):
    return re.sub(r'#\w+', '', text)

def strip_spetial_chars(text):
    return text.strip(' ,-\n\t\r*%"“”=:•^~`')

def proprocess(input_path='../TwitterAPI/data/', output_path='./data/tweets/', 
               keywords=[], exclude_keywords=[], possible_sensitive=False, 
               content_set=set(), annotation_id_set=set(), VERBOSE=False):
    
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    tweet_cnt = 0
    for json_file in get_files_in_dir(input_path):
        for exclude_keyword in exclude_keywords:
            if exclude_keyword in json_file:
                print("Skipping: ", json_file)
                continue
        
        if len(keywords) != 0:
            keyword_check = True
            for keyword in keywords:
                if keyword not in json_file:
                    keyword_check = False
                    break
            if not keyword_check:
                continue
        
        if VERBOSE:
            print("Processing: ", json_file)
        json_obj = read_json(json_file)

        for tweet in json_obj:
            tweet_cnt += 1

            new_tweet = {}
            if possible_sensitive and not tweet['possibly_sensitive']:
                continue
            new_tweet['id'] = tweet['id']
            print('\n=================================================')
            tweet_text = tweet['text']
            print(f"tweet_text before normalization: {tweet_text}")
            tweet_text = tweet['text'].strip().replace('\n', ' ').replace('\r', ' ')
            tweet_text = remove_url(tweet_text)
            tweet_text = remove_mentions(tweet_text)
            tweet_text = remove_tags(tweet_text)
            tweet_text = strip_spetial_chars(tweet_text)
            tweet_text = normalization_pipeline([tweet_text])[0]
            print(f"tweet_text after normalization: {tweet_text}")

            if (new_tweet['id'] not in annotation_id_set) and (tweet_text in content_set):
                continue
            else:
                content_set.add(tweet_text)
                new_tweet['contents'] = tweet_text
                save_json(new_tweet, output_path + '%s.json' % new_tweet['id'])
    print("Total tweets: ", tweet_cnt)
    print("Unique tweets: ", len(content_set))
    return content_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process queried tweets in json.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    keywords = args.keywords
    exclude_keywords= args.exclude_keywords

    input_path = args.input_path
    output_path = args.output_path
    
    proprocess(input_path = args.input_path, 
            output_path = output_path, 
            keywords = keywords, 
            exclude_keywords = exclude_keywords,
            possible_sensitive = False)

    print("Number of unique tweets in the corpus: ", len(set(get_files_in_dir(output_path))))
    