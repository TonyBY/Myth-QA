from typing import List, Optional
import numpy as np
import os
from tqdm import tqdm
import transformers

from pygaggle.qa.base import Answer, Ground_True_Answers, Context
from Evaluator.reader_evaluator import MultiAnswerReaderEvaluator

from Helper.utils import save_json, read_json, read_jsonl, get_raw, logger, init_logger
from StanceMining.stance_miner import EntityClaimStanceMiner, YesNoClaimStanceMiner
from IR.bm25 import BM25_Retriever
from IR.dpr import DPR_Retriever

import argparse

def define_dsearch_args(parser):
    parser.add_argument('--annotation-path', 
                        type=str, 
                        default='../data/annotations/TweetMythQA.jsonl', 
                        help='TweetMythQA annotation path.')
    
    parser.add_argument('--extrinsic-reader-result-path', 
                        type=str, 
                        default='../data/results/run1/qa/merged_dpr_reader_results_top5.jsonl', 
                        help='Path to a multiple answer prediction dataset with predicted results for extrinsic controversial stance mining evaluation.')

    parser.add_argument('--textual-corpus-dir', 
                    type=str, 
                    default='../data/processed_tweets/',
                    help='Processed tweets directory.')

    parser.add_argument('--top-k-retrievals',
                        type=int,
                        nargs='+',
                        default=[10, 100, 1000],
                        help="Different number of relevant stance evidence to retrieve when doing stance mining for each claim.")

    parser.add_argument('--top-e-stance-evidence',
                        type=int,
                        nargs='+',
                        default=[1, 5, 10, 100],
                        help="Different number of stance evidence to use when calculating stance retrival scores.")

    parser.add_argument('--yes-no-only',
                        action='store_true',
                        help='Skip evaluation for entity questions.')

    parser.add_argument('--entity-only',
                        action='store_true',
                        help='Skip evaluation for yes-no questins.')

    parser.add_argument('--metrics-types',
                        type=str,
                        nargs='+',
                        default=['ANS', 'BLEU', 'ROUGE', 'EXACT'],
                        help="Evaluation metrics type list.")

    parser.add_argument('--simple-retriever-names',
                        type=str,
                        nargs='+',
                        default=['bm25', 'dpr'],
                        help="Evaluate controversial stance mining with different tweet retriever.")

    parser.add_argument('--simple-stance-detector-names',
                        type=str,
                        nargs='+',
                        default=['roberta', 'bart', 'albert', 'bert', 'deberta'],
                        help="Evaluate controversial stance mining with different stance detector.")

    parser.add_argument('--intrinsic',
                        action='store_true',
                        help='Do intrinsic evaluation. Golden context will be used.')

class StanceMinerEvaluator():
    def __init__(self,
                 entity_claim_stances_data_items: Optional[List[dict]]=None, 
                 yes_no_claim_stances_data_items: Optional[List[dict]]=None, 
                 corpus_processed_dir: str = '../data/processed_tweets/') -> None:

        self.corpus_processed_dir = corpus_processed_dir
        self.eq_data_items = entity_claim_stances_data_items
        self.yn_data_items = yes_no_claim_stances_data_items

    def evaluate_entity_question_stances(self, 
                                         metrics_types: List[str]=['ANS', 'ROUGE', 'BLEU'], 
                                         top_e: int=1) -> dict:
        score_dict = {}
        f1_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}
        recall_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}
        precision_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}

        for i in tqdm (range(len(self.eq_data_items)), desc="Evaluating entity stance miner ..."):
            if self.eq_data_items[i]['question_type'] != 'entity':
                continue
            prediction_objs = []
            for pred in self.eq_data_items[i]['prediction'].keys():
                supporting_evidence = self.eq_data_items[i]['prediction'][pred]['supporting']
                refuting_evidence = self.eq_data_items[i]['prediction'][pred]['refuting']
                prediction_objs.append(Answer(text=pred, 
                                            supporting_evidence=supporting_evidence,
                                            refuting_evidence=refuting_evidence))
                    
            ground_truth_answers = self.eq_data_items[i]['answers'].keys()
            ground_truth_with_evidence_objs = []
            for ans in ground_truth_answers:
                gta = Ground_True_Answers(ans)
                gta.supporting_evidence = self.get_stance_tweet_list(self.eq_data_items[i], ans, "supporting")
                gta.refuting_evidence = self.get_stance_tweet_list(self.eq_data_items[i], ans, "refuting")
                gta.neutral_evidence = []
                ground_truth_with_evidence_objs.append(gta)
            
            for metrics_type in metrics_types:
                precision, recall, f1 = MultiAnswerReaderEvaluator.stance_mining_f1_score(prediction_objs, 
                                                                ground_truth_with_evidence_objs, type=metrics_type, top_e=top_e)
                
                f1_scores[metrics_type][str(topk)].append(f1)
                recall_scores[metrics_type][str(topk)].append(recall)
                precision_scores[metrics_type][str(topk)].append(precision)

        score_dict["F1"] = f1_scores
        score_dict["Recall"] = recall_scores
        score_dict["Precision"] = precision_scores
        return score_dict

    def evaluate_yes_no_question_stances(self, topk: int, 
                                         metrics_types: List[str]=["ANS", "BLEU", "ROUGE", "EXACT"], 
                                         top_e: int=1) -> dict:
        score_dict = {}
        f1_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}
        recall_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}
        precision_scores = {metrics_type: {str(topk): []} for metrics_type in metrics_types}

        for i in tqdm (range(len(self.yn_data_items)), desc="Evaluating yes-no stance miner ..."):
            if self.yn_data_items[i]['question_type'] != 'yes-no' or 'prediction' not in self.yn_data_items[i]:
                continue
            topk_prediction = self.yn_data_items[i]['prediction']
            ground_truth_answers = YesNoClaimStanceMiner.get_ground_true_answers_from_stance_annotations(self.yn_data_items[i])
            prediction_objs = []
            ground_truth_with_evidence_objs = []
            for pred in topk_prediction[f'top{topk}'].keys():
                evidence = topk_prediction[f'top{topk}'][pred]["evidence"]
                context = Context(text=evidence,
                                title="")
                prediction_objs.append(Answer(text=pred, 
                                            context=context,
                                            score=topk_prediction[f'top{topk}'][pred]["score"]))
            for ans in ground_truth_answers:
                gta = Ground_True_Answers(ans)
                gta.supporting_evidence = self.get_stance_tweet_list(self.yn_data_items[i], 
                                                                     ans, "supporting")
                gta.refuting_evidence = self.get_stance_tweet_list(self.yn_data_items[i], 
                                                                   ans, "refuting")
                gta.neutral_evidence = []
                ground_truth_with_evidence_objs.append(gta)
            
            for metrics_type in metrics_types:
                precision, recall, f1 = MultiAnswerReaderEvaluator.F1_score(prediction_objs, 
                                                                    ground_truth_with_evidence_objs, 
                                                                    type=metrics_type, only_supporting_evidence=True, top_e=top_e)

                f1_scores[metrics_type][str(topk)].append(f1)
                recall_scores[metrics_type][str(topk)].append(recall)
                precision_scores[metrics_type][str(topk)].append(precision)
              
        score_dict["F1"] = f1_scores
        score_dict["Recall"] = recall_scores
        score_dict["Precision"] = precision_scores
        return score_dict

    def get_stance_tweet_list(self, item: dict, ans: str, stance: str) -> List[str]:
        if item['question_type'] == "entity":
                stance_tweet_ids = item['answers'][ans][stance]
        else:
            stance_tweet_ids = item['answers'][stance]
        if stance_tweet_ids == []:
            return []
        else:
            stance_tweet_text_list = []
            for tweet_id in stance_tweet_ids:
                if tweet_id != "None" and tweet_id != "":
                    stance_tweet_text_list.append(get_raw(tweet_id, corpus_dir=self.corpus_processed_dir))
            return stance_tweet_text_list

def visualize_stance_mining_scores(scores_dict: dict, k: int) -> None:
    metrics_list = list(scores_dict.keys())
    if type(scores_dict['F1']) == list:
        metrics_type = "stance_mining"
        simple_score_dict = {metrics: {metrics_type: {k: float}} for metrics in metrics_list}
        f1 = np.mean(np.array(scores_dict['F1'])) * 100.
        recall = np.mean(np.array(scores_dict['Recall'])) * 100.
        precision = np.mean(np.array(scores_dict['Precision'])) * 100.

        simple_score_dict['F1'][metrics_type][k] = f1
        simple_score_dict['Recall'][metrics_type][k] = recall
        simple_score_dict['Precision'][metrics_type][k] = precision

        logger.info(f'Target Context: Top{k} tweets\tF1_{metrics_type}: {f1}')
        logger.info(f'Target Context: Top{k} tweets\tRecall_{metrics_type}: {recall}')
        logger.info(f'Target Context: Top{k} tweets\tPrecision_{metrics_type}: {precision}')
        
        print(f'Target Context: Top{k} tweets\tF1_{metrics_type}: {f1}')
        print(f'Target Context: Top{k} tweets\tRecall_{metrics_type}: {recall}')
        print(f'Target Context: Top{k} tweets\tPrecision_{metrics_type}: {precision}')
    else:
        simple_score_dict = {metrics: {metrics_type: {k: float} for metrics_type in scores_dict[metrics].keys()} for metrics in metrics_list}
        for metrics_type in scores_dict['F1'].keys():
            f1 = np.mean(np.array(scores_dict['F1'][metrics_type][str(k)])) * 100.
            recall = np.mean(np.array(scores_dict['Recall'][metrics_type][str(k)])) * 100.
            precision = np.mean(np.array(scores_dict['Precision'][metrics_type][str(k)])) * 100.
            
            simple_score_dict['F1'][metrics_type][k] = f1
            simple_score_dict['Recall'][metrics_type][k] = recall
            simple_score_dict['Precision'][metrics_type][k] = precision
            
            logger.info(f'Context: Top{k} tweets\tF1_{metrics_type}: {f1}')
            logger.info(f'Context: Top{k} tweets\tRecall_{metrics_type}: {recall}')
            logger.info(f'Context: Top{k} tweets\tPrecision_{metrics_type}: {precision}')

            print(f'Context: Top{k} tweets\tF1_{metrics_type}: {f1}')
            print(f'Context: Top{k} tweets\tRecall_{metrics_type}: {recall}')
            print(f'Context: Top{k} tweets\tPrecision_{metrics_type}: {precision}')
    return simple_score_dict

def get_intrinsic_data_for_stance_mining(annotation_jsonl_path: str) -> List[dict]:
    data_items = read_jsonl(annotation_jsonl_path)
    intrinsic_data_for_stance_mining = []
    for item in data_items:
        if item['question_type'] == 'entity':
            item['prediction'] = {ans: {} for ans in item['answers'].keys()}
        intrinsic_data_for_stance_mining.append(item)
    return intrinsic_data_for_stance_mining


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    corpus_processed_dir = args.textual_corpus_dir
    run_name = 'run1'

    INTRINSIC = args.intrinsic

    if INTRINSIC:
        logger.info("Start intrinsic experiments...")
        annotation_jsonl_path = args.annotation_path
        machine_reader_results = get_intrinsic_data_for_stance_mining(annotation_jsonl_path)
        run_name = run_name + '_intrinsic'
    else:
        logger.info("Start extrinsic experiments...")
        machine_reader_results_path = f'../data/results/{run_name}/qa/merged_dpr_reader_results_top5.jsonl'
        machine_reader_results = read_jsonl(machine_reader_results_path)

    topk_list = args.top_k_retrievals
    top_e_list = args.top_e_stance_evidence
    simple_retriever_names = args.simple_retriever_names
    simple_stance_detector_names = args.simple_stance_detector_names
    metrics_types = args.metrics_types
    
    log_file = f'../data/results/{run_name}/stance_mining/logfile.log'
    init_logger(verbose=False, log_file=log_file)
    transformers.logging.set_verbosity_error()

    for topk in topk_list:
        print(f"\n================== topk: {topk} ==================================")
        for top_e in top_e_list:
            print(f"\n================== top_e: {top_e} ==================================")
            for simple_retriever_name in simple_retriever_names:
                print(f"\n############## {simple_retriever_name} ######################")
                logger.info(f'simple_retriever_name: {simple_retriever_name}')
                if simple_retriever_name == 'bm25':
                    index_path = '../data/index/sparse_term_frequency_embedding'
                    retriever =  BM25_Retriever(k1 = 1.6,
                                                b = 0.75,
                                                index_path = index_path,
                                                processed_tweets_dir = corpus_processed_dir)
                elif simple_retriever_name == 'dpr':
                    logger.info(f"Using DPR retriever...")
                    index_path = "../data/index/dindex-sample-dpr-multi"
                    query_encoder_name = "facebook/dpr-question_encoder-multiset-base"
                    retriever = DPR_Retriever(query_encoder_name=query_encoder_name, 
                                                index_path=index_path, 
                                                processed_tweets_dir=corpus_processed_dir)
                else:
                    raise Exception(f"Unknown retriever name: {simple_retriever_name}")
                    
                for simple_stance_detector_name in simple_stance_detector_names:
                    print(f"\n====================== {simple_stance_detector_name} ======================")
                    logger.info(f'simple_stance_detector_name: {simple_stance_detector_name}')
                    if simple_stance_detector_name == 'roberta':
                        stance_detector_name = 'roberta-large-mnli'
                    elif simple_stance_detector_name == 'bart':
                        stance_detector_name = 'facebook/bart-large-mnli'
                    elif simple_stance_detector_name == 'albert':
                        stance_detector_name = 'anirudh21/albert-large-v2-finetuned-mnli'
                    elif simple_stance_detector_name == 'bert':
                        stance_detector_name = 'madlag/bert-large-uncased-mnli'
                    elif simple_stance_detector_name == 'deberta':
                        stance_detector_name = 'microsoft/deberta-large-mnli'
                    else:
                        raise Exception(f"Unknown model name: {simple_stance_detector_name}")

                    entity_claim_retrievals_path = f'../data/results/{run_name}/stance_mining/entity_claim_retrievals_{simple_retriever_name}_{topk}_{simple_stance_detector_name}.jsonl'
                    entity_claim_stances_path = f'../data/results/{run_name}/stance_mining/entity_claim_stances_{simple_retriever_name}_{topk}_{simple_stance_detector_name}.jsonl'
                    entity_claim_stance_mining_scores_path = f'../data/results/{run_name}/stance_mining/entity_claim_stance_mining_scores_{simple_retriever_name}_{topk}_{simple_stance_detector_name}_top-e{top_e}.json'
                    simple_entity_claim_stance_mining_score_dict_path = f'../data/results/{run_name}/stance_mining/simple_entity_claim_stance_mining_score_dict_{simple_retriever_name}_{topk}_{simple_stance_detector_name}_top-e{top_e}.json'

                    if args.yes_no_only:
                        pass
                    else:
                        logger.info("Start evaluating eneity stance miner resutls...")
                        if not os.path.exists(entity_claim_stances_path):
                            entity_calim_stance_miner = EntityClaimStanceMiner(retriever, 
                                                                            stance_detector_name, topk=topk,
                                                                            entity_claim_retrievals_path=entity_claim_retrievals_path,
                                                                            entity_claim_stances_path=entity_claim_stances_path)
                            
                            entity_claim_stances_data_items = entity_calim_stance_miner.stance_classification(machine_reader_results)
                        else:
                            entity_claim_stances_data_items = read_jsonl(entity_claim_stances_path)
                        
                        if not os.path.exists(entity_claim_stance_mining_scores_path) or True:
                            stanc_miner_evaluator = StanceMinerEvaluator(entity_claim_stances_data_items=entity_claim_stances_data_items)
                            score_dict = stanc_miner_evaluator.evaluate_entity_question_stances(metrics_types=metrics_types, top_e=top_e)
                            save_json(score_dict, entity_claim_stance_mining_scores_path)
                        else:
                            score_dict = read_json(entity_claim_stance_mining_scores_path)
                        print("\n************** Entity claim stance minging results: **********************")
                        simple_entity_claim_stance_mining_score_dict =  visualize_stance_mining_scores(score_dict, topk)
                        save_json(simple_entity_claim_stance_mining_score_dict, simple_entity_claim_stance_mining_score_dict_path)
                        logger.info("Done")

                    yes_no_claim_retrievals_path = entity_claim_stances_path
                    yes_no_claim_stances_path = f'../data/results/{run_name}/stance_mining/yes_no_claim_stances_{simple_retriever_name}_{topk}_{simple_stance_detector_name}.jsonl'
                    yes_no_claim_stance_mining_scores_path = f'../data/results/{run_name}/stance_mining/yes_no_claim_stance_mining_scores_{simple_retriever_name}_{topk}_{simple_stance_detector_name}_top-e{top_e}.json'
                    simple_yes_no_claim_stance_mining_score_dict_path = f'../data/results/{run_name}/stance_mining/simple_yes_no_claim_stance_mining_score_dict_{simple_retriever_name}_{topk}_{simple_stance_detector_name}_top-e{top_e}.json'

                    if args.entity_only:
                        pass
                    else:
                        logger.info("Start evaluating yes-no stance miner...")
                        if not os.path.exists(yes_no_claim_stances_path):
                            yes_no_calim_stance_miner = YesNoClaimStanceMiner(retriever, 
                                                                            stance_detector_name, topk=topk,
                                                                            yes_no_claim_retrievals_path=entity_claim_retrievals_path,
                                                                            yes_no_claim_stances_path=yes_no_claim_stances_path)
                            
                            if INTRINSIC:
                                yes_no_calim_stance_miner.golden_stance_tweet_prepare(machine_reader_results)
                            else:
                                yes_no_calim_stance_miner.stance_tweet_retrieval(machine_reader_results)
                            yes_no_claim_stances_data_items = yes_no_calim_stance_miner.stance_classification(machine_reader_results)
                        else:
                            yes_no_claim_stances_data_items = read_jsonl(yes_no_claim_stances_path)
                        
                        if not os.path.exists(yes_no_claim_stance_mining_scores_path) or True:
                            stance_miner_evaluator = StanceMinerEvaluator(yes_no_claim_stances_data_items=yes_no_claim_stances_data_items)
                            score_dict = stance_miner_evaluator.evaluate_yes_no_question_stances(topk, metrics_types=metrics_types, top_e=top_e)
                            save_json(score_dict, yes_no_claim_stance_mining_scores_path)
                        else:
                            score_dict = read_json(yes_no_claim_stance_mining_scores_path)
                        logger.info(f"score_dict['F1']['{metrics_types[0]}'].keys(): {score_dict['F1'][metrics_types[0]].keys()}")
                        print("\n************** Yes-No claim stance minging results: **********************")
                        simple_yes_no_claim_stance_mining_score_dict = visualize_stance_mining_scores(score_dict, topk)
                        save_json(simple_yes_no_claim_stance_mining_score_dict, simple_yes_no_claim_stance_mining_score_dict_path)
                        logger.info("Done")
