from typing import List, Optional
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from Helper.utils import read_jsonl, save_jsonl, get_raw, logger
from .base import Retriever


class BM25_Retriever(Retriever):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 index_path: str = None,
                 processed_tweets_dir: str = None,
                 output_path: Optional[str] = None):
        self.k1 = k1
        self.b = b
        self.corpus_index_dir = index_path
        self.corpus_processed_dir = processed_tweets_dir
        self.output_path = output_path
    
    def retrieve_tweets_for_each_question(self, items: List[dict], topk=1000) -> List[dict]:
        for i in tqdm (range(len(items)), desc="retrieving for every question..."):
            items[i]["contexts"] = []
            question = items[i]["question"]
            tweet_dict_list = self.retrieve(question, topk=topk)
            items[i]["contexts"].extend(tweet_dict_list)
        logger.info("Done")
        if self.output_path != None:
            save_jsonl(items, self.output_path)
            logger.info(f"Retrieval results have been saved to: {self.output_path}")
        return items

    def retrieve(self, query: str, topk: int=1000) -> List[dict]:
        searcher = LuceneSearcher(self.corpus_index_dir)
        searcher.set_bm25(k1=0.9, b=0.4)
        hits = searcher.search(query, k=topk,
                            remove_dups=True)
        tweet_dict_list = []
        for j in range(len(hits)):
            tweet_dict = {}
            tweet_dict["docid"] = str(hits[j].docid)
            tweet_dict["score"] = str(hits[j].score)
            tweet_dict["text"] = get_raw(hits[j].docid, self.corpus_processed_dir)
            tweet_dict_list.append(tweet_dict)
        return tweet_dict_list


if __name__ == "__main__":
    annotation_jsonl_path = "../data/annotations/Multi-answer Questions-COVID_data.jsonl"
    corpus_index_dir = "../data/index/sparse_term_frequency_embedding"
    corpus_processed_dir = "../data/temp_preprocessed_tweets/"
    output_path = "../data/retrieved_tweets/lucene_top_retrievals.jsonl"
    k1 = 1.6
    b = 0.75
    
    bm25_retriever = BM25_Retriever(k1=k1, b=b, index_path=corpus_index_dir, 
                                    processed_tweets_dir=corpus_processed_dir, 
                                    output_path=output_path)
    
    topk=1000
    items = read_jsonl(annotation_jsonl_path)
    retrieved_items = bm25_retriever.retrieve_tweets_for_each_question(items, topk=topk)
