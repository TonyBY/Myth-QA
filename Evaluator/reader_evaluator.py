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

from typing import List, Optional, Dict
from tqdm import tqdm
import os
from rouge_score import rouge_scorer, scoring
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import transformers
import gc

from pygaggle.data.retrieval import RetrievalExample
from pygaggle.qa.base import  Question, Answer, Context, Context, Ground_True_Answers, Reader
# from pygaggle.qa.span_selection import SpanSelection
from pygaggle.model.evaluate import ReaderEvaluator

from QA.t5_reader import T5Reader
from QA.dpr_reader import MultiAnswerDprReader
from QA.dpr_span_selection import parse_span_selection_rules

import argparse

def define_dsearch_args(parser):
    parser.add_argument('--annotation-path', 
                        type=str, 
                        default='../data/annotations/TweetMythQA.jsonl', 
                        help='TweetMythQA annotation path.')

    parser.add_argument('--intrinsic-retrieval-path', 
                        type=str, 
                        default=f'../data/annotations/TweetMythQA_with_gold_context.jsonl', 
                        help='Path to a multiple answer prediction dataset with gold context for intrinsic evaluation.')
    
    parser.add_argument('--extrinsic-retrieval-path', 
                        type=str, 
                        default='../data/results/run1/retrieval/bm25_top1000_retrievals.jsonl', 
                        help='Path to a multiple answer prediction dataset with retrieved context for extrinsic evaluation.')

    parser.add_argument('--textual-corpus-dir', 
                    type=str, 
                    default='../data/processed_tweets/',
                    help='Processed tweets directory.')

    parser.add_argument('--context-size-list',
                        type=int,
                        nargs='+',
                        default=[10, 100, 1000],
                        help="Evaluate reader with different context size.")

    parser.add_argument('--answer-number-list',
                        type=int,
                        nargs='+',
                        default=[1, 5, 10],
                        help="Evaluate reader with different number of distinct predictions.")

    parser.add_argument('--intrinsic',
                        action='store_true',
                        help='Do intrinsic evaluation. Golden context will be used.')

class MultiAnswerReaderEvaluator(ReaderEvaluator):
    """
    This class contains functions to evaluate results of multi-answer machine readers.

    Paras:
    exmples: A list of RetrievalExample instance. It contains the resuls of tweet retrieval by questions.
    topk_em: a list of integers representing differen number of retrievals to use when doing mache reading comprehension.
    mr_predictions: a list of dictionaries where stores the results of the machine reader.
    metrics_types: a list of metrics names indicating the metrics to use when evaluating the machine reader's results.

    scores_dict: a dictionary that contains the scores of the results of the machine reader under different experiment settings 
                 and evaluation metrics.
    """
    def evaluate(
        self,
        examples: List[RetrievalExample],
        topk_em: List[int] = [50],
        mr_predictions: Optional[Dict[int, List[Dict[str, str]]]] = None,
        metrics_types: List[str] = ["ROUGE"]
    ):
        scores_dict = {}
        ems = {str(setting): {str(k): [] for k in topk_em} for setting in self.reader.span_selection_rules}
        f1_scores = {metrics_type: {str(setting): {str(k): [] for k in topk_em} for setting in self.reader.span_selection_rules} for metrics_type in metrics_types}
        recall_scores = {metrics_type: {str(setting): {str(k): [] for k in topk_em} for setting in self.reader.span_selection_rules} for metrics_type in metrics_types}
        precision_scores = {metrics_type: {str(setting): {str(k): [] for k in topk_em} for setting in self.reader.span_selection_rules} for metrics_type in metrics_types}

        for example in tqdm(examples, desc="Evaluating reader's results..."):
            answers = self.reader.predict(example.question, example.contexts, topk_em)
            ground_truth_answers_with_evidence = example.ground_truth_answers_with_evidence
            ground_truth_answers = [ans.text for ans in ground_truth_answers_with_evidence]

            topk_prediction = {str(setting): {} for setting in self.reader.span_selection_rules}

            for setting in self.reader.span_selection_rules:
                for k in topk_em:
                    try:
                        best_answer = answers[str(setting)][k][0].text
                    except:
                        best_answer = ""

                    top_answers = {ans.text: {"evidence": ans.context.text, "score": ans.score} for ans in answers[str(setting)][k]} 
                
                    for metrics_type in metrics_types:
                        precision, recall, f1 = MultiAnswerReaderEvaluator.F1_score(answers[str(setting)][k], 
                                                                                    ground_truth_answers_with_evidence, 
                                                                                    type=metrics_type)

                        f1_scores[metrics_type][str(setting)][str(k)].append(f1)
                        recall_scores[metrics_type][str(setting)][str(k)].append(recall)
                        precision_scores[metrics_type][str(setting)][str(k)].append(precision)

                    em_hit = max([MultiAnswerReaderEvaluator.exact_match_score(best_answer, ga) for ga in ground_truth_answers])
                    ems[str(setting)][str(k)].append(em_hit)

                    topk_prediction[f'{str(setting)}'][f'top{k}'] = top_answers

            scores_dict["EM"] = ems
            scores_dict["F1"] = f1_scores
            scores_dict["Recall"] = recall_scores
            scores_dict["Precision"] = precision_scores
            
            if mr_predictions is not None:
                mr_predictions.append({
                    'question': example.question.text,
                    'answers': list(ground_truth_answers),
                    'prediction': topk_prediction,
                })

        return scores_dict

    @staticmethod
    def F1_score(prediction: List[Answer], 
                 ground_truth_with_evidence: List[RetrievalExample.ground_truth_answers_with_evidence], 
                 type: str='ROUGE', only_supporting_evidence: bool=False, top_e: int=1) -> float:
        similarity_scores = []
        for pred in prediction:
            for gt in ground_truth_with_evidence:
                if MultiAnswerReaderEvaluator._normalize_answer(pred.text) == MultiAnswerReaderEvaluator._normalize_answer(gt.text):
                    print(f"Find match!")
                    if only_supporting_evidence:
                        ground_true_evidence = gt.supporting_evidence
                        logger.info(f"pred.context.text: {pred.context.text}")
                        similarity_scores.append(MultiAnswerReaderEvaluator.get_provenance_similarity_score(pred.context.text, ground_true_evidence, type=type, top_e=top_e))
                    else:
                        ground_true_evidence = gt.supporting_evidence + gt.refuting_evidence
                        similarity_scores.append(MultiAnswerReaderEvaluator.get_provenance_similarity_score([pred.context.text], ground_true_evidence, type=type, top_e=top_e))
                    continue
        if len(prediction) == 0:
            precision = 0
        else:
            precision = sum(similarity_scores) / len(prediction)

        if len(ground_truth_with_evidence) == 0:
            recall = 0
        else:
            recall = sum(similarity_scores) / len(ground_truth_with_evidence)

        f1 = 0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def stance_mining_f1_score(prediction: List[Answer], 
                               ground_truth_with_evidence: List[RetrievalExample.ground_truth_answers_with_evidence], 
                               type: str='ROUGE', top_e: int=1) -> float:
        if len(prediction) == 0:
            precision = 0
            recall = 0
        else:
            similarity_scores = []
            for pred in prediction:
                for gt in ground_truth_with_evidence:
                    if MultiAnswerReaderEvaluator._normalize_answer(pred.text) == MultiAnswerReaderEvaluator._normalize_answer(gt.text):
                        if type == "ANS":
                            similarity_scores.append(1.0)
                        else:
                            similarity_scores.append(MultiAnswerReaderEvaluator.get_stance_similarity_score(pred, gt, type=type, top_e=top_e))
                        continue

            precision = sum(similarity_scores) / len(prediction)
            recall = sum(similarity_scores) / len(ground_truth_with_evidence)

        f1 = 0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def get_provenance_similarity_score(prediction_provenance: List[str], 
                                        gold_references: List[str], 
                                        type: str="ROUGE",
                                        n_gram: int=4,
                                        normalize: bool = False,
                                        top_e:int=1):
        
        prediction_provenance = prediction_provenance[: min(top_e, len(prediction_provenance))]
        if normalize:
            logger.info("Normalizing evidence...")
            prediction_provenance = normalization_pipeline(prediction_provenance)
            gold_references = normalization_pipeline(gold_references)
            logger.info("Done.")
        
        if type == "ANS":
            return 1.0
        elif type == "ROUGE":
            ROUGE_type = "rougeL"
            if gold_references == []:
                return 1.0
            max_score = 0.0
            scorer = rouge_scorer.RougeScorer([ROUGE_type], use_stemmer=True)
            for pred in prediction_provenance:
                for ref in gold_references:
                    score = scorer.score(ref, pred)[ROUGE_type].recall
                    if score > max_score:
                        max_score = score
            return max_score
        elif type == "BLEU":
            if gold_references == []:
                return 1.0
            max_score = 0.0
            for pred in prediction_provenance:
                for ref in gold_references:
                    candidate_tokens = [token for token in pred.split()]
                    gold_tokens = [token for token in ref.split()]

                    candidate_tokens = [x for x in candidate_tokens if len(x.strip()) > 0]
                    gold_tokens = [x for x in gold_tokens if len(x.strip()) > 0]
                    # # The default BLEU calculates a score for up to 4-grams using uniform weights (this is called BLEU-4)
                    # weights = [0.25, 0.25, 0.25, 0.25]
                    weights = [1.0 / n_gram for _ in range(n_gram)]

                    if len(gold_tokens) < n_gram:
                        # lower order ngrams
                        weights = [1.0 / len(gold_tokens) for _ in range(len(gold_tokens))]

                    chencherry = nltk.translate.bleu_score.SmoothingFunction()

                    BLEUscore = nltk.translate.bleu_score.sentence_bleu(
                        [candidate_tokens], gold_tokens, 
                        weights=weights, 
                        smoothing_function=chencherry.method1
                    )
                    if BLEUscore > max_score:
                        max_score = BLEUscore
            return max_score
        elif type == "EXACT":
            if gold_references == []:
                return 1.0
            for pred in prediction_provenance:
                for ref in gold_references:
                    if pred == ref:
                        return 1.0
            return 0.0

        else:
            raise("Error: Illegal metircs type: %s" % type + "\n only support: 'ANS', 'ROUGE', and 'BLEU'")

    @staticmethod
    def get_stance_similarity_score(pred: Answer, gt: RetrievalExample.ground_truth_answers_with_evidence, type:str ="ROUGE", top_e:int =1):
        if type == "ANS":
            return 1
        
        if len(gt.supporting_evidence) == 0 and len(pred.supporting_evidence) == 0:
            supporting_score = 1
        elif len(gt.supporting_evidence) == 0 and len(pred.supporting_evidence) != 0:
            supporting_score = 0
        elif len(gt.supporting_evidence) != 0 and len(pred.supporting_evidence) == 0:
            supporting_score = 0
        else:
            supporting_score = MultiAnswerReaderEvaluator.get_provenance_similarity_score(pred.supporting_evidence, gt.supporting_evidence, type=type, top_e=top_e)
        
        if len(gt.refuting_evidence) == 0 and len(pred.refuting_evidence) == 0:
            refuting_score = 1
        elif len(gt.refuting_evidence) == 0 and len(pred.refuting_evidence) != 0:
            refuting_score = 0
        elif len(gt.refuting_evidence) != 0 and len(pred.refuting_evidence) == 0:
            refuting_score = 0
        else:
            refuting_score = MultiAnswerReaderEvaluator.get_provenance_similarity_score(pred.refuting_evidence, gt.refuting_evidence, type=type, top_e=top_e)
        
        if len(gt.supporting_evidence) != 0 and len(gt.refuting_evidence) != 0:
            stance_similarity_score = (supporting_score + refuting_score) / 2.0
        elif len(gt.supporting_evidence) != 0 and len(gt.refuting_evidence) == 0:
            stance_similarity_score = supporting_score
        elif len(gt.supporting_evidence) == 0 and len(gt.refuting_evidence) != 0:
            stance_similarity_score = refuting_score
        else: # len(gt.supporting_evidence) == 0 and len(gt.refuting_evidence) == 0:
            logger.warning(f"This answer does not have stance annotation!: {gt.text}")
            stance_similarity_score = 1
        return stance_similarity_score

def init_t5_reader_from_str(model: str, num_answers: int=10, batch_size: int=32, device=None):
    logger.info('Loading Reader Model and Tokenizer')
    tokenizer = model
    return T5Reader(model, tokenizer, num_answers=num_answers, batch_size=batch_size, device=device)

def get_stance_tweet_list(item: dict, ans: str, stance: str, corpus_processed_dir: str) -> List[str]:
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
                stance_tweet_text_list.append(get_raw(tweet_id, corpus_dir=corpus_processed_dir))
        return stance_tweet_text_list

def get_retrival_example_list(items: List[dict], max_topk_passages: int, corpus_processed_dir: str, INTRINSIC: bool=False):
    if INTRINSIC:
        CONTEXT = 'gold_contexts'
    else:
        CONTEXT = 'contexts'
    examples = []
    for i in tqdm(range(len(items)), desc="Building RetrivalExamples..."):
        item = items[i]
        if item['question_type'] == "yes-no":
            continue

        if CONTEXT not in item:
            break
        topk_contexts = sorted(item[CONTEXT], reverse=True, key=lambda context: float(context["score"]))[
                        : max_topk_passages]
        texts = list(map(lambda context: Context(text=context['text'].split('\n', 1)[-1].replace('""', '"'),
                                                 title="",
                                                 score=float(context["score"])),
                         topk_contexts))

        ground_true_answers_list = []
        for ans in item['answers'].keys():
            gta = Ground_True_Answers(ans)

            gta.supporting_evidence = get_stance_tweet_list(item, ans, "supporting", corpus_processed_dir)
            gta.refuting_evidence = get_stance_tweet_list(item, ans, "refuting", corpus_processed_dir)
            gta.neutral_evidence = get_stance_tweet_list(item, ans, "neutral", corpus_processed_dir)

            ground_true_answers_list.append(gta)

        examples.append(
            RetrievalExample(
                question=Question(text=item['question']),
                contexts=texts,
                ground_truth_answers=item['answers'].keys(),
                ground_truth_answers_with_evidence=ground_true_answers_list
            )
        )
    return examples

def eval_mr_reader(examples: RetrievalExample, reader: Reader, 
                    evaluator: MultiAnswerReaderEvaluator, 
                    topk_em: List[int] = [50], 
                    metrics_types: List[str] = ["ANS", "BLEU", "ROUGE", "EXACT"]):
    mr_predictions = []
    scores_dict = evaluator.evaluate(examples, topk_em, mr_predictions, 
                                        metrics_types=metrics_types)

    logger.info('Reader completed')
    return mr_predictions, scores_dict

def visualize_reader_scores_dict(scores_dict: dict, num_answers: int):
    metrics_types = list(scores_dict['F1'].keys())
    span_selection_rules = list(scores_dict['F1'][metrics_types[0]].keys())
    topk_em = list(scores_dict['F1'][metrics_types[0]][span_selection_rules[0]].keys())

    simple_score_dict = {metrics_name: {metrics_type: {str(setting): {k: float for k in topk_em} for setting in span_selection_rules} for metrics_type in metrics_types} for metrics_name in ['F1', 'Recall', 'Precision']}
    simple_score_dict['EM'] = {setting: {k: float for k in topk_em} for setting in span_selection_rules}

    best_k = topk_em[-1]
    best_setting = span_selection_rules[0]
    best_recall = 0
    for setting in span_selection_rules:
        print("========================================")
        print(f'Setting: {str(setting)}') 
        logger.info(f'Setting: {str(setting)}')
        for k in topk_em:
            em = np.mean(np.array(scores_dict['EM'][str(setting)][k])) * 100.
            simple_score_dict['EM'][str(setting)][k] = em
            logger.info(f'Context: Top{k} tweets\tEM: {em}')
            for metrics_type in metrics_types:
                f1 = np.mean(np.array(scores_dict['F1'][metrics_type][str(setting)][k])) * 100.
                recall = np.mean(np.array(scores_dict['Recall'][metrics_type][str(setting)][k])) * 100.
                precision = np.mean(np.array(scores_dict['Precision'][metrics_type][str(setting)][k])) * 100.
                
                simple_score_dict['F1'][metrics_type][str(setting)][k] = f1
                simple_score_dict['Recall'][metrics_type][str(setting)][k] = recall
                simple_score_dict['Precision'][metrics_type][str(setting)][k] = precision
                
                logger.info(f'Context: Top{k} tweets\tF1_{metrics_type}@{num_answers}: {f1}')
                logger.info(f'Context: Top{k} tweets\tRecall_{metrics_type}@{num_answers}: {recall}')
                logger.info(f'Context: Top{k} tweets\tPrecision_{metrics_type}@{num_answers}: {precision}')

                print(f'Context: Top{k} tweets\tF1_{metrics_type}@{num_answers}: {f1}')
                print(f'Context: Top{k} tweets\tRecall_{metrics_type}@{num_answers}: {recall}')
                print(f'Context: Top{k} tweets\tPrecision_{metrics_type}@{num_answers}: {precision}')

                if recall > best_recall:
                    best_recall = recall
                    best_k = k
                    best_setting = setting
    return best_k, str(best_setting), simple_score_dict


def init_dpr_reader_from_str(model: str, num_spans: int = 10, 
                         settings: List[str] = ["dpr", "dprfusion_1.0_0.55"], 
                         max_answer_length: int = 5, num_spans_per_passage: int = 5, 
                         batch_size: int = 32, device: str = 'cuda:0'):
    logger.info('Loading Reader Model and Tokenizer')
    tokenizer = model
    span_selection_rules = [parse_span_selection_rules(setting) for setting in settings]
    return MultiAnswerDprReader(model,
                                tokenizer,
                                span_selection_rules,
                                num_spans,
                                max_answer_length,
                                num_spans_per_passage,
                                batch_size,
                                device)

def add_machine_reader_results_to_general_retrieval_results(mr_predictions: List[dict], general_retrievals_items: List[dict], 
                                                       span_selection_setting: str="dpr", topk: int=50):
    j = 0
    for i in range(len(general_retrievals_items)):
        if general_retrievals_items[i]['question_type'] == 'yes-no':
            continue
        assert general_retrievals_items[i]["question"] == mr_predictions[j]["question"]
        general_retrievals_items[i]['prediction'] = mr_predictions[j]["prediction"][span_selection_setting][f"top{topk}"]
        j += 1
    return general_retrievals_items

def get_gold_contex(data_jsonl: List[dict], processed_tweet_dir: str) -> List[dict]:
    data_jsonl_with_gold_context = []
    for row in data_jsonl:
        question = row['question']
        print(f'question: {question}')
        question_type = row['question_type']
        if question_type == 'entity':
            gold_context_ids = []
            gold_context = []
            answers = row['answers']
            for ans in answers:
                print(f'answer: {ans}')
                supporting_tweet_ids = row['answers'][ans]['supporting']
                refuting_tweet_ids = row['answers'][ans]['refuting']
                gold_context_ids.extend(supporting_tweet_ids)
                gold_context_ids.extend(refuting_tweet_ids)
            for context_id in gold_context_ids:
                gold_context.append({'docid': context_id, 
                                     'score': 1.0, 
                                     "text": get_raw(context_id, corpus_dir=processed_tweet_dir)})
        else:
            gold_context_ids = []
            gold_context = []
            
            supporting_tweet_ids = row['answers']['supporting']
            refuting_tweet_ids = row['answers']['refuting']
            gold_context_ids.extend(supporting_tweet_ids)
            gold_context_ids.extend(refuting_tweet_ids)
                
            for context_id in gold_context_ids:
                gold_context.append({'docid': context_id, 
                                     'score': 1.0, 
                                     "text": get_raw(context_id, corpus_dir=processed_tweet_dir)})
        row['gold_contexts'] = gold_context
        data_jsonl_with_gold_context.append(row)
    return data_jsonl_with_gold_context

def get_golden_retrieval_result(input_jsonl_path: str, output_jsonl_path: str, corpus_processed_dir:str) -> None:
    data_jsonl = read_jsonl(input_jsonl_path)
    data_jsonl_with_gold_context = get_gold_contex(data_jsonl, corpus_processed_dir)
    save_jsonl(data_jsonl_with_gold_context, output_jsonl_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    annotation_jsonl_path = args.annotation_path
    corpus_processed_dir = args.textual_corpus_dir
    run_name = 'run1'

    INTRINSIC = args.intrinsic

    if INTRINSIC:
        print('============ Intrinsic Evaluation ==============')
        retrieval_file_jsonl_path = args.intrinsic_retrieval_path
        run_name = run_name + '_intrinsic' 
        if not os.path.exists(retrieval_file_jsonl_path):
            get_golden_retrieval_result(annotation_jsonl_path, retrieval_file_jsonl_path, corpus_processed_dir)
    else:
        print('============ Extrinsic Evaluation ==============')
        retrieval_file_jsonl_path = args.extrinsic_retrieval_path
        if not os.path.exists(retrieval_file_jsonl_path):
            raise Exception(f"File does not exist: {retrieval_file_jsonl_path}. Please use MythQA/Evaluator/etriever_evaluator.py or MythQA/IR/bm25.py or MythQA/IR/dpr.py to generate first.")
    
    log_file = f'../data/results/{run_name}/qa/machine_readers_logfile.log'
    init_logger(verbose=False, log_file=log_file)
    transformers.logging.set_verbosity_error()

    gc.enable()
    torch.cuda.empty_cache() 
    SEED = 1024
    ## Setting device for PyTorch to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(SEED)
        logger.info(f"There are {torch.cuda.device_count()} GPU(s) avaiable")
        logger.info(f"We will be using the following GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU available using the CPU instead")
        device =  torch.device("cpu")
    
    batch_size = 32
    num_answers_list =args.answer_number_list
    max_topk_passages = 100
    
    topk_em = args.context_size_list
    metrics_types = ["ANS", "BLEU", "ROUGE", "EXACT"]

    logger.info('Loading the Retrieval File')
    general_retrievals_jsonl = read_jsonl(retrieval_file_jsonl_path)
    logger.info('Done.')
    logger.info('Building example list for readers...')
    retrival_example_list = get_retrival_example_list(general_retrievals_jsonl, max_topk_passages, 
                                                  corpus_processed_dir, INTRINSIC=INTRINSIC)
    logger.info('Done.')

    for num_answers in num_answers_list:
        print(f'\n======================= num_answers: {num_answers} ===================================')
        print('\n####################### T5 Reader #################################')
        pretrained_t5_reader_name = "valhalla/t5-base-qa-qg-hl"
        t5_reader = init_t5_reader_from_str(pretrained_t5_reader_name, num_answers=num_answers, batch_size=batch_size, device=device)
        t5_reader_evaluator = MultiAnswerReaderEvaluator(t5_reader)
        t5_predictions, t5_scores_dict = eval_mr_reader(retrival_example_list, t5_reader, t5_reader_evaluator, 
                                                        topk_em=topk_em, metrics_types=metrics_types)
        t5_best_k, t5_best_setting, simple_t5_score_dict = visualize_reader_scores_dict(t5_scores_dict, num_answers)
        logger.info(f"t5_best_k: {t5_best_k}")
        logger.info(f"t5_best_setting: {t5_best_setting}")
        print(f"t5_best_k: {t5_best_k}")
        print(f"t5_best_setting: {t5_best_setting}")
        t5_predictions_jsonl_path = f"../data/results/{run_name}/qa/t5_reader_results.jsonl"
        save_jsonl(t5_predictions, t5_predictions_jsonl_path)

        merged_t5_reader_results = add_machine_reader_results_to_general_retrieval_results(t5_predictions, 
                                                                                        general_retrievals_jsonl,
                                                                                        span_selection_setting=t5_best_setting, 
                                                                                        topk=t5_best_k)
        merged_t5_predictions_jsonl_path = f"../data/results/{run_name}/qa/merged_t5_reader_results_top{num_answers}.jsonl"
        save_jsonl(merged_t5_reader_results, merged_t5_predictions_jsonl_path)

        print('\n####################### DPR Reader #################################')
        num_spans_per_passage=10
        pretrained_dpr_reader_name = "facebook/dpr-reader-single-nq-base"
        span_selection_rules = ["dpr", "dprfusion_1.0_0.55"]

        dpr_reader = init_dpr_reader_from_str(pretrained_dpr_reader_name, num_spans=num_answers, 
                                            settings=span_selection_rules, 
                                            max_answer_length=5, num_spans_per_passage=1, 
                                            batch_size=32, device=device)
        dpr_reader_evaluator = MultiAnswerReaderEvaluator(dpr_reader)
        dpr_predictions, dpr_scores_dict = eval_mr_reader(retrival_example_list, dpr_reader, dpr_reader_evaluator, 
                                                        topk_em=topk_em, metrics_types=metrics_types)
        dpr_best_k, dpr_best_setting, simple_dpr_score_dict = visualize_reader_scores_dict(dpr_scores_dict, num_answers)
        logger.info(f"dpr_best_k: {dpr_best_k}")
        logger.info(f"dpr_best_setting: {dpr_best_setting}")
        print(f"dpr_best_k: {dpr_best_k}")
        print(f"dpr_best_setting: {dpr_best_setting}")

        dpr_predictions_jsonl_path = f"../data/results/{run_name}/qa/dpr_reader_results.jsonl"
        save_jsonl(dpr_predictions, dpr_predictions_jsonl_path)

        merged_dpr_reader_results = add_machine_reader_results_to_general_retrieval_results(dpr_predictions, 
                                                                                        general_retrievals_jsonl,
                                                                                        span_selection_setting=dpr_best_setting, 
                                                                                        topk=dpr_best_k)
        
        merged_dpr_predictions_jsonl_path = f"../data/results/{run_name}/qa/merged_dpr_reader_results_top{num_answers}.jsonl"
        save_jsonl(merged_dpr_reader_results, merged_dpr_predictions_jsonl_path)
