try:
    from Helper.utils import read_jsonl, get_raw, logger, init_logger
except:
    import sys
    from pprint import pprint
    print("sys.path:")
    pprint(sys.path)
    print("\nBefore running the code, make sure the project dir is in sys.path.")
    print("Otherwise, run 'export PYTHONPATH=$<your project dir>' in your ternimal first.\n")
    exit()

from typing import List
from tqdm import tqdm
import os
from IR.bm25 import BM25_Retriever
from IR.dpr import DPR_Retriever
import transformers

import argparse

def define_dsearch_args(parser):
    parser.add_argument('--annotation-path', 
                        type=str, 
                        default='../data/annotations/TweetMythQA.jsonl', 
                        help='TweetMythQA annotation path.')

    parser.add_argument('--textual-corpus-dir', 
                    type=str, 
                    default='../data/processed_tweets/',
                    help='Processed tweets directory.')

    parser.add_argument('--top-k-retrievals',
                        type=int,
                        nargs='+',
                        default=[10, 100, 500, 1000],
                        help="Different top-k retrieval numbers to evaluate.")


class RetrieverEvaluator():
    def __init__(self, corpus_processed_dir: str):
        self.corpus_processed_dir = corpus_processed_dir

    def evaluate(self, items: List[dict], k: int, EVIDENCE_FORM: str='docid') -> dict:
        entity_question_hit_scores = []
        entity_question_mhit_scores = []
        yes_no_question_hit_scores = []
        yes_no_question_mhit_scores = []
        
        for i in tqdm(range(len(items)), desc=f'Calculating HIT@{k} scores'):
            question = items[i]['question']
            logger.debug(f"question: {question}")
            question_type = items[i]['question_type']
            if EVIDENCE_FORM == 'text':
                general_retrievals = [context['text'] for context in items[i]['contexts']]
            elif EVIDENCE_FORM == 'docid':
                general_retrievals = [context['docid'] for context in items[i]['contexts']]
            else:
                raise Exception(f'Unsupported EVIDENCE_FORM: {EVIDENCE_FORM}')

            if question_type == 'entity':
                answers = items[i]['answers'].keys()
                evidence_list = []
                answer_evidence_finding_scores = []
                for ans in answers:
                    ans_evidence = []
                    for stance in ['supporting', 'refuting']:
                        for docid in items[i]['answers'][ans][stance]:
                            logger.debug(f"docid: {docid}")
                            if len(docid) <= 7 and len(docid) >= 19:
                                continue
                            if EVIDENCE_FORM == 'text':
                                evidence = get_raw(docid, corpus_dir=self.corpus_processed_dir)
                            elif EVIDENCE_FORM == 'docid':
                                evidence = docid
                            else:
                                raise Exception(f'Unsupported EVIDENCE_FORM: {EVIDENCE_FORM}')
                            ans_evidence.append(evidence)
                        
                    if ans_evidence != [] and len(set(ans_evidence).intersection(set(general_retrievals[:k]))) != 0:
                        answer_evidence_finding_scores.append(1)
                    elif ans_evidence == []:
                        logger.warning(f"This claim has no evidence: {question}|||{ans}. Skipping it.")
                        continue
                    else:
                        answer_evidence_finding_scores.append(0)
                    logger.debug(f"answer_evidence_finding_scores: {answer_evidence_finding_scores}")
                
                    evidence_list.extend(ans_evidence)
                    
                if len(set(evidence_list)) != 0:
                    hit_score = 1.0 if len(set(evidence_list).intersection(set(general_retrievals[:k]))) != 0 else 0.0
                else:
                    hit_score = 0.0
                
                if len(answer_evidence_finding_scores) != 0:
                    mhit_score = hit_score * (sum(answer_evidence_finding_scores) / len(answer_evidence_finding_scores))
                else:
                    mhit_score = 0.0
                logger.debug(f"hit_score: {hit_score}")
                logger.debug(f"mhit_score: {mhit_score}")

                entity_question_hit_scores.append(hit_score)
                entity_question_mhit_scores.append(mhit_score)

            elif question_type == 'yes-no':
                evidence_list = []
                answer_evidence_finding_scores = []
                ans_evidence = []
                for stance in ['supporting', 'refuting']:
                    for docid in items[i]['answers'][stance]:
                        if len(docid) <= 7 and len(docid) >= 19:
                            continue
                        logger.debug(f"docid: {docid}")
                        if EVIDENCE_FORM == 'text':
                            evidence = get_raw(docid, corpus_dir=self.corpus_processed_dir)
                        elif EVIDENCE_FORM == 'docid':
                            evidence = docid
                        else:
                            raise Exception(f'Unsupported EVIDENCE_FORM: {EVIDENCE_FORM}')
                        logger.debug(f"evidence: {evidence}")

                        ans_evidence.append(evidence)
                    evidence_list.extend(ans_evidence)
                    
                if ans_evidence != [] and len(set(ans_evidence).intersection(set(general_retrievals[:k]))) != 0:
                    answer_evidence_finding_scores.append(1)
                elif ans_evidence == []:
                    logger.warning(f"This claim has no evidence: {question}|||Yes. Skipping it.")
                    continue
                else:
                    answer_evidence_finding_scores.append(0)
                logger.debug(f"answer_evidence_finding_scores: {answer_evidence_finding_scores}")

                logger.debug(f"evidence_list: {evidence_list}")
                if len(set(evidence_list)) != 0:
                    hit_score = 1.0 if len(set(evidence_list).intersection(set(general_retrievals[:k]))) != 0 else 0.0
                else:
                    hit_score = 0.0
                if len(answer_evidence_finding_scores) != 0:
                    mhit_score = hit_score * (sum(answer_evidence_finding_scores) / len(answer_evidence_finding_scores))
                else:
                    mhit_score = 0.0
                    
                logger.debug(f"hit_score: {hit_score}")
                logger.debug(f"mhit_score: {mhit_score}")
                yes_no_question_hit_scores.append(hit_score)
                yes_no_question_mhit_scores.append(mhit_score)
            else:
                raise Exception(f"Unknown question type encountered: {question_type}")
        
        logger.debug(f"entity_question_hit_scores: {entity_question_hit_scores}")
        logger.debug(f"yes_no_question_hit_scores: {yes_no_question_hit_scores}")
        all_hit_scores = entity_question_hit_scores + yes_no_question_hit_scores
        all_mhit_scores = entity_question_mhit_scores + yes_no_question_mhit_scores
        
        if len(entity_question_hit_scores) == 0:
            logger.debug(f"entity_question_hit_scores: {entity_question_hit_scores}")
            avg_entity_hit_score = 0
        else:
            logger.debug(f"entity_question_hit_scores: {entity_question_hit_scores}")
            avg_entity_hit_score = sum(entity_question_hit_scores) / len(entity_question_hit_scores)
        if len(entity_question_mhit_scores) == 0:
            logger.debug(f"entity_question_mhit_scores: {entity_question_mhit_scores}")
            avg_entity_mhit_score = 0
        else:
            logger.debug(f"entity_question_mhit_scores: {entity_question_mhit_scores}")
            avg_entity_mhit_score = sum(entity_question_mhit_scores) / len(entity_question_mhit_scores)
        
        if len(yes_no_question_hit_scores) == 0:
            avg_yes_no_hit_score = 0
        else:
            avg_yes_no_hit_score = sum(yes_no_question_hit_scores) / len(yes_no_question_hit_scores)
        if len(yes_no_question_mhit_scores) == 0:
            avg_yes_no_mhit_score = 0
        else:
            avg_yes_no_mhit_score = sum(yes_no_question_mhit_scores) / len(yes_no_question_mhit_scores)
        
        logger.debug(f"all_hit_scores: {all_hit_scores}")
        avg_all_hit_score = sum(all_hit_scores) / len(all_hit_scores)
        avg_all_mhit_score = sum(all_mhit_scores) / len(all_mhit_scores)
        
        score_dict = {}
        score_dict['avg_all_hit_score'] = avg_all_hit_score
        score_dict['avg_all_mhit_score'] = avg_all_mhit_score
        score_dict['avg_entity_hit_score'] = avg_entity_hit_score
        score_dict['avg_entity_mhit_score'] = avg_entity_mhit_score
        score_dict['avg_yes_no_hit_score'] = avg_yes_no_hit_score
        score_dict['avg_yes_no_mhit_score'] = avg_yes_no_mhit_score    
        return score_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    annotation_jsonl_path = args.annotation_path
    corpus_processed_dir = args.textual_corpus_dir
    run_name = 'run1'
    
    transformers.logging.set_verbosity_error()
    log_file = f'../data/results/{run_name}/retrieval/logfile.log'
    init_logger(verbose=False, log_file=log_file)

    logger.info(f"annotation_jsonl_path: {annotation_jsonl_path}")
    logger.info(f"corpus_processed_dir: {corpus_processed_dir}")
    
    topk_list = args.top_k_retrievals
    for topk in topk_list:
        print(f"\n================== top: {topk} ==============================")
        print("\n######################## BM25 ##############################")
        logger.info('\nLoading the BM25 Retrieval File')
        bm25_retrievals_jsonl_path = f"../data/results/{run_name}/retrieval/bm25_top{max(topk_list)}_retrievals.jsonl"

        if os.path.exists(bm25_retrievals_jsonl_path):
            logger.info(f'Loading cached retrievals from: {bm25_retrievals_jsonl_path}')
            bm25_general_retrievals_jsonl = read_jsonl(bm25_retrievals_jsonl_path)
        else:
            logger.info(f"Retrieving with BM25...")
            corpus_index_dir = "../data/index/sparse_term_frequency_embedding"
            k1 = 1.6
            b = 0.75
            
            bm25_retriever = BM25_Retriever(k1=k1, b=b, index_path=corpus_index_dir, 
                                            processed_tweets_dir=corpus_processed_dir, 
                                            output_path=bm25_retrievals_jsonl_path)
            items = read_jsonl(annotation_jsonl_path)
            bm25_general_retrievals_jsonl = bm25_retriever.retrieve_tweets_for_each_question(items, topk=max(topk_list))
            logger.info("Done.")
        logger.info("Evaluating BM25...")
        bm25_evaluator = RetrieverEvaluator(corpus_processed_dir)
        bm25_evaluator_score_dict = bm25_evaluator.evaluate(bm25_general_retrievals_jsonl, topk)
        logger.info("Done.")
        logger.info(f"bm25_evaluator_score_dict: \n{bm25_evaluator_score_dict}")
        print(f"bm25_evaluator_score_dict: \n{bm25_evaluator_score_dict}")
        
        print("######################## DPR ##############################")
        dpr_retrievals_jsonl_path = f"../data/results/{run_name}/retrieval/DPR_top{max(topk_list)}_retrievals.jsonl"

        if os.path.exists(dpr_retrievals_jsonl_path):
            logger.info(f'Loading cached retrievals from: {dpr_retrievals_jsonl_path}')
            dpr_general_retrievals_jsonl = read_jsonl(dpr_retrievals_jsonl_path)
        else:
            logger.info(f"Retrieving with DPR...")
            corpus_index_dir = "../data/index/dindex-sample-dpr-multi"
            query_encoder_name = "facebook/dpr-question_encoder-multiset-base"

            dpr_retriever = DPR_Retriever(query_encoder_name=query_encoder_name, 
                                            index_path=corpus_index_dir, 
                                            processed_tweets_dir=corpus_processed_dir, 
                                            output_path=dpr_retrievals_jsonl_path)
            items = read_jsonl(annotation_jsonl_path)
            dpr_general_retrievals_jsonl = dpr_retriever.retrieve_tweets_for_each_question(items, topk=max(topk_list))
            logger.info("Done")

        logger.info("Evaluating DPR...")
        dpr_evaluator = RetrieverEvaluator(corpus_processed_dir)
        dpr_evaluator_score_dict = dpr_evaluator.evaluate(dpr_general_retrievals_jsonl, topk)
        logger.info("Done.")
        logger.info(f"dpr_evaluator_score_dict: \n{dpr_evaluator_score_dict}")
        print(f"dpr_evaluator_score_dict: \n{dpr_evaluator_score_dict}")    
