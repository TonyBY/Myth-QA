from typing import List, Optional
from tqdm import tqdm
from pyserini.search.faiss import FaissSearcher

from Helper.utils import read_jsonl, save_jsonl, get_raw_with_title, logger
from .base import Retriever


class DPR_Retriever(Retriever):
    def __init__(self,
                 index_path: str = None,
                 processed_tweets_dir: str = None,
                 output_path: Optional[str] = None,
                 query_encoder_name: str = "facebook/dpr-question_encoder-multiset-base"):
        
        self.corpus_index_dir = index_path
        self.corpus_processed_dir = processed_tweets_dir
        self.output_path = output_path
        self.query_encoder_name = query_encoder_name
    
    def retrieve_tweets_for_each_question(self, items: List[dict], topk=1000) -> List[dict]:
        for i in tqdm (range(len(items)), desc="retrieving for every question..."):
            items[i]["contexts"] = []
            question = items[i]["question"]
            tweet_dict_list = self.retrieve(question, topk=topk)
            items[i]["contexts"].extend(tweet_dict_list)
        if self.output_path != None:
            save_jsonl(items, self.output_path)
            logger.info(f"Question retrievals are saved to: {self.output_path}")
        return items

    def retrieve(self, query: str, topk: int=1000) -> List[dict]:
        searcher = FaissSearcher(self.corpus_index_dir, self.query_encoder_name)
            
        hits = searcher.search(query, k=topk,
                            threads=4,
                            return_vector=True)[1]
        tweet_dict_list = []
        for j in range(len(hits)):
            tweet_dict = {}
            tweet_dict["docid"] = str(hits[j].docid)
            tweet_dict["score"] = str(hits[j].score)
            tweet_dict["text"] = get_raw_with_title(hits[j].docid, self.corpus_processed_dir)
            tweet_dict_list.append(tweet_dict)
        return tweet_dict_list

    
if __name__ == "__main__":
    import torch
    import gc
    gc.enable()
    torch.cuda.empty_cache() 
    import transformers
    transformers.logging.set_verbosity_error()
    
    try:
         from Helper.utils import *
    except:
        import sys
        print("sys.path: \n", sys.path)
        print("\nBefore running the code, make sure the project dir is in sys.path.")
        print("Otherwise, run 'export PYTHONPATH=$<your project dir>' in your ternimal first.\n")

    annotation_jsonl_path = "../data/annotations/Multi-answer Questions-COVID_data.jsonl"
    corpus_index_dir = "../data/index/dindex-sample-dpr-multi"
    corpus_processed_dir = "../data/temp_preprocessed_tweets/"
    output_path = "../data/retrieved_tweets/DPR_top_retrievals_test.jsonl"
    query_encoder_name = "facebook/dpr-question_encoder-multiset-base"

    dpr_retriever = DPR_Retriever(query_encoder_name=query_encoder_name, 
                                    index_path=corpus_index_dir, 
                                    processed_tweets_dir=corpus_processed_dir, 
                                    output_path=output_path)
    
    topk=100
    items = read_jsonl(annotation_jsonl_path)
    retrieved_items = dpr_retriever.retrieve_tweets_for_each_question(items, topk=topk)
