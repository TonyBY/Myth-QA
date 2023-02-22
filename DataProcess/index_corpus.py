import os
from Helper.utils import make_directory

import argparse

def define_dsearch_args(parser):
    parser.add_argument('--processed-tweet-dir', 
                        type=str, 
                        default='../data/processed_tweets/', 
                        help='Dirctory saving json files of processed tweets.')

    parser.add_argument('--dense-index-path', 
                        type=str, 
                        default='../data/index/dindex-sample-dpr-multi', 
                        help='Dirctory saving densely indexed tweets.')
    
    parser.add_argument('--sparse-index-path', 
                        type=str, 
                        default='../data/index/sparse_term_frequency_embedding', 
                        help='Dirctory saving sparsly indexed tweets.')

def index_all_raw_tweets(processed_tweets_dir, index_path, 
                         query_encoder_name="facebook/dpr-ctx_encoder-multiset-base", 
                         batch_size=32, indexer_type="Dense"):
    index_path = make_directory(index_path)
    if indexer_type == "Dense":
        indexing_script = f"python -m pyserini.encode input --corpus {processed_tweets_dir} --fields text output  --embeddings {index_path} --to-faiss encoder --encoder {query_encoder_name} --fields text --batch {batch_size}"
    elif indexer_type == "Sparse":
        indexing_script = f"python -m pyserini.index.lucene --collection JsonCollection --input {processed_tweets_dir} --index {index_path} --generator DefaultLuceneDocumentGenerator --threads 1"
    else:
        raise Exception(f"Unknown indexer type: {indexer_type}")
    os.system(indexing_script)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process queried tweets in json.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    processed_tweet_dir = args.processed_tweet_dir
    dense_index_path = args.dense_index_path
    sparse_index_path = args.sparse_index_path
    
    index_all_raw_tweets(processed_tweet_dir, dense_index_path, 
                     query_encoder_name="facebook/dpr-ctx_encoder-multiset-base", batch_size=32, indexer_type="Dense")
    
    index_all_raw_tweets(processed_tweet_dir, sparse_index_path,  indexer_type="Sparse")
