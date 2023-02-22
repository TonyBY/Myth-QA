from cmath import e
import os
import time
import requests
from tqdm import tqdm
from Helper.utils import read_json, save_json
from typing import List

import argparse

def define_dsearch_args(parser):
    parser.add_argument('--tweet-id-corpus-file', 
                        type=str, 
                        default='../data/id_corpus/tweet_ids.json', 
                        help='The json file path saving all tweet ids in the corpus.')

    parser.add_argument('--root-annotation-dir', 
                        type=str, 
                        default='../data/raw/', 
                        help='Dirctory saving raw tweets.')

# To set your environment variables in your terminal run the following line:
# export 'TWEETER_ACDAMIC_BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("TWEETER_ACDAMIC_BEARER_TOKEN")

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r

def create_url(ids_str, tweet_fields, user_fields, place_fields):
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    ids = "ids=" + ids_str
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = f"https://api.twitter.com/2/tweets?{ids}&{tweet_fields}&{user_fields}&{place_fields}"
    return url

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    response_code = response.status_code
    return response_code, response

def get_all_ids_from_cached_corpus(corpus_saving_path):
    corpus_data = read_json(corpus_saving_path)
    tweet_ids = []
    for data in corpus_data:
        tweet_id = data['id']
        tweet_ids.append(tweet_id)
    return tweet_ids

def collectTweetsByIds(tweet_ids: set(), root_raw_tweet_dir: str, batch_size: int=1,
                       tweet_fields: str="tweet.fields=lang,author_id", 
                       user_fields: str="id,name,username,location", 
                       place_fields: str="id,name,country_code,full_name,country",
                       cache_interval: int=1000) -> List[dict]:
    twitter_data = []
    failed_ids = []
    output_dir = root_raw_tweet_dir + 'tweets/'
    corpus_saving_path = output_dir + 'corpus.json'
    failed_ids_saving_path = output_dir + 'failed_ids.json'

    total_tweet_ids_in_corpus = len(tweet_ids)
    pbar = tqdm(total = total_tweet_ids_in_corpus)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.path.exists(corpus_saving_path):
        cached_tweet_ids = get_all_ids_from_cached_corpus(corpus_saving_path)
        twitter_data = read_json(corpus_saving_path)
        print(f"Loading {len(cached_tweet_ids)} tweets from cached corpus at: {corpus_saving_path}")

        if os.path.exists(failed_ids_saving_path):
            failed_ids = read_json(failed_ids_saving_path)
            print(f"Loading recorded {len(failed_ids)} invalid tweet ids from: {failed_ids_saving_path}")
    else:
        cached_tweet_ids = []
        
    pbar.update(len(cached_tweet_ids))

    tweet_ids = set(tweet_ids) - set(tweet_ids).intersection(set(cached_tweet_ids)) - set(failed_ids)
    tweet_ids = list(tweet_ids)

    if len(tweet_ids) == 0 and len(failed_ids) !=0:
        tweet_ids = list(set(failed_ids))
        batch_size = 1
        cache_interval = 20
        failed_ids = []
    else:
        pbar.update(len(failed_ids))

    if tweet_ids == []:
        return twitter_data

    current_idx = 0
    STOP = False
    print(f"len(tweet_ids): {len(tweet_ids)}")
    print(f"current_idx: {current_idx}")
    print(f"batch_size: {batch_size}")
    cache_cnt = int(len(twitter_data) / cache_interval) + 1
    while not STOP:
        if len(tweet_ids[current_idx:]) >= batch_size:
            ids_str = ",".join(tweet_ids[current_idx: current_idx + batch_size])
            current_idx += batch_size
            pbar.update(batch_size)
        else:
            ids_str = ",".join(tweet_ids[current_idx:])
            pbar.update(batch_size - current_idx)
            STOP = True
        
        if ids_str == "":
            continue

        url = create_url(ids_str, tweet_fields, user_fields, place_fields)
        print("ids_str: ", ids_str)

        response_code, response = connect_to_endpoint(url)
        print("response_code: ", response_code)
        SUCCESS_CALL = False
        retry_cnt = 0
        while not SUCCESS_CALL:
            if response_code == 200:
                retry_cnt = 0
                SUCCESS_CALL = True
            elif response_code == 429:
                time.sleep(6)
                retry_cnt += 1
                response = requests.request("GET", url, auth=bearer_oauth)
                response_code = response.status_code
                if retry_cnt % 10 == 0:
                    print("response_code: ", response_code)
                if retry_cnt == 1:
                    print(f"Saving {len(twitter_data)} cached tweets to: {corpus_saving_path}")
                    save_json(twitter_data, corpus_saving_path)
            else:
                print(f"Saving {len(twitter_data)} cached tweets to: {corpus_saving_path}")
                save_json(twitter_data, corpus_saving_path)
                raise Exception("Request returned an error: {} {}".format(
                    response.status_code, response.text)
                )
        
        if 'data' not in response.json():
            failed_ids.extend(ids_str.split(','))
            print(f"Invalid tweet ids: {ids_str}")
        else:
            print(f"Received response to tweet_ids: {ids_str}.")
            for idx, tweet in enumerate(response.json()['data']):
                twitter_data.append(tweet)

        if int(len(twitter_data) / cache_interval) == cache_cnt:
            print(f"Saving {len(twitter_data)} cached tweets to: {corpus_saving_path}")
            save_json(twitter_data, corpus_saving_path)
            cache_cnt += 1
            time.sleep(3)

        if len(failed_ids) % 10 == 0:
            print(f"Saving {len(failed_ids)} failed tweets to: {failed_ids_saving_path}")
            save_json(failed_ids, failed_ids_saving_path)
            time.sleep(3)

        time.sleep(1)
        
    pbar.close()

    print("API requests are done. Saving to %s ..." % corpus_saving_path)
    save_json(twitter_data, corpus_saving_path)

    print("Saving failed_ids to %s ..." % failed_ids_saving_path)
    save_json(failed_ids, failed_ids_saving_path)

    print(f"len(failed_ids): {len(failed_ids)}")

    return twitter_data

def main(tweet_id_corpus_file, root_annotation_dir):
    uncached_tweet_ids = read_json(tweet_id_corpus_file)
    twitter_data = collectTweetsByIds(uncached_tweet_ids, root_annotation_dir, batch_size=50,
                                   tweet_fields="tweet.fields=lang,author_id", 
                                   user_fields="user.fields=id,name,username,location", 
                                   place_fields="place.fields=id,name,country_code,full_name,country")
    print(f"len(twitter_data): {len(twitter_data)}")
    return twitter_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    tweet_id_corpus_file = args.tweet_id_corpus_file
    root_annotation_dir = args.root_annotation_dir

    main(tweet_id_corpus_file, root_annotation_dir)
