import argparse
import os
import transformers
from pprint import pprint
import time

from IR.bm25 import BM25_Retriever
from IR.dpr import DPR_Retriever
from StanceMining.stance_miner import EntityClaimStanceMiner, YesNoClaimStanceMiner
from Evaluator.retriever_evaluator import RetrieverEvaluator
from Evaluator.reader_evaluator import MultiAnswerReaderEvaluator, init_t5_reader_from_str, init_dpr_reader_from_str, \
                                       eval_mr_reader, get_retrival_example_list, \
                                       add_machine_reader_results_to_general_retrieval_results, visualize_reader_scores_dict
from Evaluator.stance_miner_evaluator import StanceMinerEvaluator, visualize_stance_mining_scores
from Helper.utils import read_jsonl, save_jsonl, read_json, save_json, make_directory, logger, init_logger, set_device

RETRIEVER_CHOICES = ('dpr', 'bm25')
READER_CHOICES = ('dpr_reader', 't5')
QUESTION_TYPE_CHOICES = ('entity', 'yes-no', 'not_given')

def define_dsearch_args(parser):
    # Demo mode
    parser.add_argument('--demo', 
                        type=str, 
                        default='', 
                        help='demo question string.')

    parser.add_argument('--questoin-type', type=str, 
                        choices=QUESTION_TYPE_CHOICES, 
                        default='not_given',
                        help="Question type. default value: not-given. Choice range: ('entity', 'yes-no', 'not_given')")
    # General paras
    parser.add_argument('--dataset', 
                        type=str, 
                        default='./data/annotations/Multi-answer Questions-COVID_data.jsonl',
                        help='Dataset path.')

    parser.add_argument('--processed-corpus-dir', 
                        type=str, 
                        default='./data/processed_tweets/',
                        help='Processed tweets directory.')

    parser.add_argument('--results-dir',
                        type=str,
                        default='./data/results/end2end/')

    parser.add_argument('--batch-size', 
                        type=int, 
                        default=32, 
                        help='Batch size.')

    parser.add_argument('--use-cache',
                        action='store_true',
                        help='Flag to used cached intermediate results if available.')

    parser.add_argument('--cache-results', 
                        action='store_true',
                        help='Flag to cache intermediate results.')

    # Retriever paras
    parser.add_argument('--index-path', type=str, 
                        default='./data/index/sparse_term_frequency_embedding',
                        help='Indexed corpus path.')

    parser.add_argument('--retriever',
                        type=str,
                        default='bm25',
                        choices=RETRIEVER_CHOICES,
                        help="Retriever choice. Choices are: bm25 and dpr.")

    parser.add_argument('--number-of-retrievals',
                        type=int,
                        default=1000,
                        help='Number of results returned by the retriever. Default value: 1000')

    parser.add_argument('--k1',
                        type=float,
                        default = '1.6',
                        help='A parameter of BM25 retriever. Default value: 1.6')

    parser.add_argument('--b',
                        type=float,
                        default = '0.75',
                        help='A parameter of BM25 retriever. Default value: 0.75')

    parser.add_argument('--query-encoder-name',
                        type=str,
                        default = 'facebook/dpr-question_encoder-multiset-base',
                        help="Pretrained query encoder's name of DPR retriever. Default: facebook/dpr-question_encoder-multiset-base")
    
    # Reader paras
    parser.add_argument('--num-asnwers',
                        type=int,
                        default=5,
                        help="Maximum number of distinct answers to return by the reader.")

    parser.add_argument('--reader',
                        type=str,
                        default='dpr_reader',
                        choices=READER_CHOICES,
                        help="Reader choice. Choices are: dpr_reader and t5.")

    parser.add_argument('--pretrained-t5-reader-model',
                        type=str,
                        default = 'valhalla/t5-base-qa-qg-hl',
                        help="Pretrained T5 reader model's name. Default: 'valhalla/t5-base-qa-qg-hl'")

    parser.add_argument('--pretrained-dpr-reader-model',
                        type=str,
                        default = 'facebook/dpr-reader-single-nq-base',
                        help="Pretrained DPR reader model's name. Default: 'facebook/dpr-reader-single-nq-base'")

    parser.add_argument('--dpr-settings',
                        type=str,
                        nargs='+',
                        default=['dprfusion_1.0_0.55'],
                        help="Method name(s) of span selection for the DPR reader. Default value: dpr.")

    # Stance Miner paras
    parser.add_argument('--pretrained-nli-model',
                        type=str,
                        default='roberta-large-mnli',
                        help='Pretrained NLI model name.')

def get_question_type(question: str) -> str:
    """
    Given a question, return the type of question: entity or yes-no.
    Paras: 
        ques: quesiton text
    return:
        ques_type: either "entity" or "yes-no"
    """

    question_clean = question.strip().lower()
    if question_clean.startswith("do") or question_clean.startswith("did") or question_clean.startswith("does") \
        or question_clean.startswith("is") or question_clean.startswith("was") \
        or question_clean.startswith("are") or question_clean.startswith("were") \
        or question_clean.startswith("can") or question_clean.startswith("could") \
        or question_clean.startswith("have") or question_clean.startswith("had") \
        or question_clean.startswith("will")  or question_clean.startswith("wold") \
        or question_clean.startswith("can") or question_clean.startswith("could"):
            return "binary"
    return "entity"

def visualize_demo_results(demo_result: dict, k: int = 3):
    demo_dict = {'question': demo_result['question'],
                 'question_type': demo_result['question_type'],
                 'prediction': {}}

    logger.info(f"demo_result.keys(): {demo_result.keys()}")
    predictions = demo_result['prediction']
    for pre in predictions.keys():
        demo_dict['prediction'][pre] = {'supporting': predictions[pre]['supporting'][:k],
                                        'refuting': predictions[pre]['refuting'][:k]}
    logger.info(f"demo results: {demo_dict}")
    pprint(demo_dict)
    return demo_dict


if __name__ == "__main__":
    start_tiem = time.time()
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description='Search a Faiss index.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    results_dir = make_directory(args.results_dir)

    if args.demo != '':
        question = args.demo
        question_type = args.questoin_type
        EVAL = False
        if question_type == 'not_given':
            question_type = get_question_type(question)
        data_items = [{'question': args.demo, 
                       'question_type': question_type}]
        if question_type == 'entity':
            data_items[0]['answers'] = {'demo_ans': {'supporting': [], 'refuting': [], 'neutral': []}}
        else:
            data_items[0]['answers'] = {'supporting': [], 'refuting': [], 'neutral': []}
        data_name = question
        results_dir = make_directory(results_dir + "demos/")
    else:
        data_items = read_jsonl(args.dataset)
        data_name = args.dataset.split('/')[-1].split('.')[0]
        results_dir = make_directory(results_dir + "evals/")
        EVAL = True

   

    indexed_corpus_path = args.index_path
    processed_corpus_dir = make_directory(args.processed_corpus_dir)
    number_of_retrievals = args.number_of_retrievals
    num_answers = args.num_asnwers
    batch_size = args.batch_size
    use_cached_results = args.use_cache

    retriever_choice = args.retriever
    reader_choice = args.reader
    stance_detector_name = args.pretrained_nli_model

    results_dir = make_directory(results_dir + f"{data_name}_{retriever_choice}_{reader_choice}_{stance_detector_name.split('/')[-1]}_{number_of_retrievals}_{num_answers}/")
    score_dir = make_directory(results_dir + "scores/")

    log_file = f'{results_dir}logfile.log'
    init_logger(verbose=False, log_file=log_file)

    # Intermediate results paths
    retrieved_items_path = f"{results_dir}QuestionRetrievals.jsonl"
    merged_reader_results_path = f"{results_dir}MergedReaderPredictions.jsonl"
    entity_claim_stances_data_items_path = f"{results_dir}EntityClaimStances.jsonl" 
    yes_no_claim_stances_data_items_path = f"{results_dir}CompleteClaimStances.jsonl" 
    
    # Intermediate scores paths
    retriever_evaluator_score_dict_path = f"{score_dir}RetrieverScores.json"
    simple_reader_score_dict_path = f"{score_dir}ReaderScores.json"
    simple_entity_claim_stance_mining_score_dict_path = f"{score_dir}EntityClaimStancesScores.json"
    simple_yes_no_claim_stance_mining_score_dict_path = f"{score_dir}YesNoClaimStancesScores.json" 

    logger.info("Start doing question retrieval...")
    if retriever_choice == 'bm25':
        retriever = BM25_Retriever(k1=args.k1, b=args.b, 
                                index_path=indexed_corpus_path, 
                                processed_tweets_dir=processed_corpus_dir)
    elif retriever_choice == 'dpr':
        retriever = DPR_Retriever(query_encoder_name=args.query_encoder_name, 
                                index_path=indexed_corpus_path, 
                                processed_tweets_dir=processed_corpus_dir)
    else:
        raise Exception(f"Error: Not supported retriever choice: {retriever_choice}. Supported retriever choices are: {RETRIEVER_CHOICES}")
    if use_cached_results and os.path.exists(retrieved_items_path):
        retrieved_items = read_jsonl(retrieved_items_path)
    else:
        retrieved_items = retriever.retrieve_tweets_for_each_question(data_items, topk=number_of_retrievals)
        retriever_evaluator = RetrieverEvaluator(processed_corpus_dir)
        retriever_evaluator_score_dict = retriever_evaluator.evaluate(retrieved_items, number_of_retrievals)

        if args.cache_results:
            save_jsonl(retrieved_items, retrieved_items_path)
            save_json(retriever_evaluator_score_dict, retriever_evaluator_score_dict_path)
    logger.info("Done")

    logger.info("Start doing reading comprehension ...")
    if use_cached_results and os.path.exists(merged_reader_results_path):
        merged_reader_results = read_jsonl(merged_reader_results_path)
    else:
        if reader_choice == 't5':
            pretrained_t5_reader_name = args.pretrained_t5_reader_model
            reader = init_t5_reader_from_str(pretrained_t5_reader_name, num_answers=num_answers, batch_size=batch_size)
        elif reader_choice == 'dpr_reader':
            logger.info(f"args.dpr_settings: {args.dpr_settings}")
            logger.info(f"type(args.dpr_settings): {type(args.dpr_settings)}")
            pretrained_dpr_reader_name = args.pretrained_dpr_reader_model
            reader = init_dpr_reader_from_str(pretrained_dpr_reader_name, num_spans=num_answers, 
                                            settings=args.dpr_settings, 
                                            max_answer_length=5, num_spans_per_passage=1, 
                                            batch_size=batch_size, device=set_device())
        else:
            raise Exception(f"Error: Not supported reader choice: {reader_choice}. Supported reader choices are: {READER_CHOICES}")
        reader_evaluator = MultiAnswerReaderEvaluator(reader)
        retrival_example_list = get_retrival_example_list(retrieved_items, number_of_retrievals, processed_corpus_dir)
        reader_predictions, reader_scores_dict = eval_mr_reader(retrival_example_list, reader, reader_evaluator, 
                                                                        topk_em=[number_of_retrievals])
        best_k, best_setting, simple_reader_score_dict = visualize_reader_scores_dict(reader_scores_dict, num_answers)
        merged_reader_results = add_machine_reader_results_to_general_retrieval_results(reader_predictions, 
                                                                                        retrieved_items,
                                                                                        span_selection_setting=best_setting, 
                                                                                        topk=best_k) 
        if args.cache_results:
            save_json(simple_reader_score_dict, simple_reader_score_dict_path)
            save_jsonl(merged_reader_results, merged_reader_results_path)
    logger.info("Done")

    logger.info("Start doing entity claim stance mining...")
    if use_cached_results and os.path.exists(entity_claim_stances_data_items_path):
        entity_claim_stances_data_items = read_jsonl(entity_claim_stances_data_items_path)
    else:
        entity_calim_stance_miner = EntityClaimStanceMiner(retriever, stance_detector_name, topk=number_of_retrievals)
        entity_claim_stances_data_items = entity_calim_stance_miner.stance_classification(merged_reader_results)
        if args.cache_results:
            save_jsonl(entity_claim_stances_data_items, entity_claim_stances_data_items_path)
    logger.info("Done")

    logger.info("Start doing yes-no claim stance mining...")
    if use_cached_results and os.path.exists(yes_no_claim_stances_data_items_path):
        yes_no_claim_stances_data_items = read_jsonl(yes_no_claim_stances_data_items_path)
    else:
        yes_no_calim_stance_miner = YesNoClaimStanceMiner(retriever, stance_detector_name, topk=number_of_retrievals)
        yes_no_calim_stance_miner.stance_tweet_retrieval(entity_claim_stances_data_items)
        yes_no_claim_stances_data_items = yes_no_calim_stance_miner.stance_classification(entity_claim_stances_data_items)
        if args.cache_results:
            save_jsonl(yes_no_claim_stances_data_items, yes_no_claim_stances_data_items_path)
    logger.info("Done")

    if EVAL:
        logger.info("Start evaluating entity claim stance_mining...")
        if use_cached_results and os.path.exists(simple_entity_claim_stance_mining_score_dict_path):
            simple_entity_claim_stance_mining_score_dict = read_json(simple_entity_claim_stance_mining_score_dict_path)
        else:
            stanc_miner_evaluator = StanceMinerEvaluator(entity_claim_stances_data_items=entity_claim_stances_data_items,
                                                         corpus_processed_dir=processed_corpus_dir)
            entity_claim_stance_mining_score_dict = stanc_miner_evaluator.evaluate_entity_question_stances()
            simple_entity_claim_stance_mining_score_dict =  visualize_stance_mining_scores(entity_claim_stance_mining_score_dict, number_of_retrievals)    
            save_json(simple_entity_claim_stance_mining_score_dict, simple_entity_claim_stance_mining_score_dict_path)
        
        logger.info("Done")
        logger.info("Start evaluating yes-no claim stance_mining...")
        if use_cached_results and os.path.exists(simple_yes_no_claim_stance_mining_score_dict_path):
            simple_yes_no_claim_stance_mining_score_dict = read_json(simple_yes_no_claim_stance_mining_score_dict_path)
        else:
            stance_miner_evaluator = StanceMinerEvaluator(yes_no_claim_stances_data_items=yes_no_claim_stances_data_items,
                                                          corpus_processed_dir=processed_corpus_dir)
            yes_no_claim_stance_mining_score_dict = stance_miner_evaluator.evaluate_yes_no_question_stances(number_of_retrievals)
            simple_yes_no_claim_stance_mining_score_dict = visualize_stance_mining_scores(yes_no_claim_stance_mining_score_dict, number_of_retrievals)
            save_json(simple_yes_no_claim_stance_mining_score_dict, simple_yes_no_claim_stance_mining_score_dict_path)
        logger.info("Done")
    
    if args.demo != '':
        demo_preds = visualize_demo_results(yes_no_claim_stances_data_items[0], k=1)
    
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_tiem} s")