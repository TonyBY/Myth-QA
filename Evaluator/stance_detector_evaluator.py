try:
    from Helper.utils import read_jsonl, get_raw, build_claim, logger, init_logger
except:
    import sys
    from pprint import pprint
    print("sys.path:")
    pprint(sys.path)
    print("\nBefore running the code, make sure the project dir is in sys.path.")
    print("Otherwise, run 'export PYTHONPATH=$<your project dir>' in your ternimal first.\n")
    exit()

from typing import List
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import seaborn as sns
import transformers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

from StanceDetection.stance_detector import StanceDetector

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

class StanceDetectorEvaluator():
    def __init__(self,   
                 stance_detector: StanceDetector,
                 corpus_processed_dir: str = "../data/processed_tweets/"):

        self.corpus_processed_dir = corpus_processed_dir
        self.stance_detector = stance_detector
        self.stance_detection_df = self.stance_detector.data_df
    
    def show_distribution(self, df, column):
        supporint_num = len(df.loc[df[column].str.lower() == 'supporting'])
        refuting_num = len(df.loc[df[column].str.lower() == 'refuting'])
        neutral_num = len(df.loc[df[column].str.lower() == 'neutral'])
        print("Corpus size: ", len(df))
        print("Label distribution: ")
        print("\t Supporting: ", supporint_num)
        print("\t Refuting: ", refuting_num)
        print("\t Neutral: ", neutral_num)
        names='Supporting', 'Refuting', 'Neutral'
        size_of_groups = [supporint_num, refuting_num, neutral_num]
        plt.pie(size_of_groups, labels=names, labeldistance=1.15)
        plt.show()
        plt.savefig(f"{self.stance_detector.output_path[:-4]}_{column}_distribution.png")
        plt.clf()
        
    def plot_cf_matrix(self, y_true, y_pred, 
                       labels=['supporting', 'refuting', 'neutral']):
        cf_matrix = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=None)

        cf_matrix_df = pd.DataFrame(cf_matrix,
                            index = ['Supporting','Refuting','Neutral'], 
                            columns = ['Supporting','Refuting','Neutral'])

        #Plotting the confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(cf_matrix_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Predicted Labels')
        plt.xlabel('Golden Labels')
        plt.show()
        plt.savefig(f"{self.stance_detector.output_path[:-4]}_CM.png")
        plt.clf()

        # Calculate metrics for each label, and find their unweighted mean. 
        # This does not take label imbalance into account
        with open(f"{self.stance_detector.output_path[:-4]}_scores.txt", 'w') as f:
            f.write(f"F1 for Supporing, Refuting, and Neutral: {[f1_score(y_true, y_pred, labels=['supporting'], average='macro'), f1_score(y_true, y_pred, labels=['refuting'], average='macro'), f1_score(y_true, y_pred, labels=['neutral'], average='macro')]}")
            f.write(f"\nMacro F1:  {f1_score(y_true, y_pred, average='macro')}")
            f.write(f"\nRecall for Supporing, Refuting, and Neutral: {[recall_score(y_true, y_pred, labels=['supporting'], average='macro'), recall_score(y_true, y_pred, labels=['refuting'], average='macro'), recall_score(y_true, y_pred, labels=['neutral'], average='macro')]}")
            f.write(f"\nMacro Recall: {recall_score(y_true, y_pred, average='macro')}")
            f.write(f"\nMicro Recall: {recall_score(y_true, y_pred, average='micro')}")
            
            f.write(f"\nPrecision for Supporing, Refuting, and Neutral: {[precision_score(y_true, y_pred, labels=['supporting'], average='macro'), precision_score(y_true, y_pred, labels=['refuting'], average='macro'), precision_score(y_true, y_pred, labels=['neutral'], average='macro')]}")
            f.write(f"\nMacro Precision: {precision_score(y_true, y_pred, average='macro')}")
            
            f.write(f"\nAccuracy Score: {accuracy_score(y_true, y_pred)}")
            
            f.write(f"\nAccuracy for Supporing, Refuting, and Neutral: {cf_matrix.diagonal()/cf_matrix.sum(axis=1)}")
            
        print("F1 for Supporing, Refuting, and Neutral: ", [f1_score(y_true, y_pred, labels=["supporting"], average='macro'), 
                                                            f1_score(y_true, y_pred, labels=["refuting"], average='macro'),
                                                            f1_score(y_true, y_pred, labels=["neutral"], average='macro')])
        print("Macro F1: ", f1_score(y_true, y_pred, average='macro'))
        print("Recall for Supporing, Refuting, and Neutral: ", [recall_score(y_true, y_pred, labels=["supporting"], average='macro'), 
                                                                recall_score(y_true, y_pred, labels=["refuting"], average='macro'),
                                                                recall_score(y_true, y_pred, labels=["neutral"], average='macro')])
        print("Macro Recall: ", recall_score(y_true, y_pred, average='macro'))
        print("Micro Recall: ", recall_score(y_true, y_pred, average='micro'))
        
        print("Precision for Supporing, Refuting, and Neutral: ", [precision_score(y_true, y_pred, labels=["supporting"], average='macro'), 
                                                                precision_score(y_true, y_pred, labels=["refuting"], average='macro'),
                                                                precision_score(y_true, y_pred, labels=["neutral"], average='macro')])
        print("Macro Precision: ", precision_score(y_true, y_pred, average='macro'))
        
        
        print("Accuracy Score: ", accuracy_score(y_true, y_pred))
        
        print("Accuracy for Supporing, Refuting, and Neutral: ", cf_matrix.diagonal()/cf_matrix.sum(axis=1))
    
        
    def show_cf(self, df, label_column_name, prediction_column_name):
        df = df.drop(df.index[pd.isna(df[label_column_name])], inplace=False)
        
        for index, row in df.iterrows():
            if type(row[label_column_name]) != str or type(row[prediction_column_name]) != str:
                print(row)

        y_true = np.array(df[label_column_name])
        y_pred = np.array(df[prediction_column_name])
        
        valid_index = []
        for i in range(len(y_true)):
            if type(y_true[i]) != str or type(y_pred[i]) != str:
                print(i, y_true[i], y_pred[i])
                continue
            valid_index.append(i)
        y_true = y_true[valid_index]
        y_pred = y_pred[valid_index]

        self.plot_cf_matrix(y_true, y_pred, labels=['supporting', 'refuting', 'neutral'])
    
    def evaluate(self):
        self.stance_detection_df = self.stance_detector.do_stance_detection()

        label_column = "stance"
        prediction_column = "pred_label_text"
        print(f"type(self.stance_detection_df): {type(self.stance_detection_df)}")
        self.show_distribution(self.stance_detection_df, label_column)
        self.show_distribution(self.stance_detection_df, prediction_column)
        self.show_cf(self.stance_detection_df, label_column, prediction_column)

def polulate_columns_for_stance_detection_dataset(columns:dict, premise: str, stance_tweets: dict, 
                                                    stance: str, corpus_processed_dir: str):
    if stance_tweets[stance] == []:
        return

    for hypo in stance_tweets[stance]:
        if hypo == "" or hypo == None:
            continue
        columns["premise"].append(premise)
        columns["hypothesis_ids"].append(hypo)
        columns["hypothesis_text"].append(get_raw(hypo, corpus_dir=corpus_processed_dir))
        columns["stance"].append(stance)
        
def construct_stance_detection_dataset(items: List[dict], corpus_processed_dir: str) -> pd.DataFrame:
    columns = {"premise": [], "hypothesis_ids": [], "hypothesis_text": [], "stance": []}
    for i in tqdm (range(len(items)), desc="Constrcting dataset for stance detection..."):
        question = items[i]["question"]
        question_type = items[i]["question_type"]
        answers = items[i]["answers"]
        if question_type == "entity":
            for ans in answers.keys():
                items[i]['answers'][ans]['target_context'] = []               
                premise = build_claim(question, ans)
                
                for stance in items[i]['answers'][ans].keys():
                    polulate_columns_for_stance_detection_dataset(columns, premise, items[i]['answers'][ans], 
                                                                        stance, corpus_processed_dir)
                
        elif question_type == "yes-no":
            premise = build_claim(question, 'yes')
            for stance in items[i]['answers'].keys():
                polulate_columns_for_stance_detection_dataset(columns, premise, items[i]['answers'], 
                                                                    stance, corpus_processed_dir)
        else:
            raise Exception("Unknown question type: ", question_type)
            
    return pd.DataFrame.from_dict(columns).drop_duplicates(subset=['premise', 'hypothesis_ids'], keep='first', ignore_index=True, inplace=False)

def print_stats_of_stance_detection_df(df):
    print("#Supporting: ", len(df[df.stance=="supporting"]))
    print("#Refuting: ", len(df[df.stance=="refuting"]))
    print("#Neutral: ", len(df[df.stance=="neutral"]))
    print("Total annotations: ",  len(df))
    logger.info(f"#Supporting: {len(df[df.stance=='supporting'])}")
    logger.info(f"#Refuting: {len(df[df.stance=='refuting'])}")
    logger.info(f"#Neutral: {len(df[df.stance=='neutral'])}")
    logger.info(f"Total annotations: {len(df)}")

    cnt_contra = 0
    for premise in df.premise.unique():
        stances = set(df[df["premise"]==premise].stance)
        if set(["supporting", "refuting"]).issubset(stances):
            cnt_contra += 1
    print("#Unique Premise: ", len(df.premise.unique()))
    print("#Controversial Premises: ", cnt_contra)
    logger.info(f"#Unique Premise: {len(df.premise.unique())}")
    logger.info(f"#Controversial Premises: {cnt_contra}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_dsearch_args(parser)
    args = parser.parse_args()

    annotation_jsonl_path = args.annotation_path
    corpus_processed_dir = args.textual_corpus_dir
    run_name = 'run1'

    log_file = f'../data/results/{run_name}/stance_detection/logfile.log'
    init_logger(verbose=False, log_file=log_file)
    transformers.logging.set_verbosity_error()

    batch_size = 32

    anno_data_jsonl = read_jsonl(annotation_jsonl_path)
    stance_detection_df = construct_stance_detection_dataset(anno_data_jsonl, corpus_processed_dir)

    simple_data_name = annotation_jsonl_path.split('/')[-1].split('.')[0]
    stance_detection_df.to_csv(f"../data/results/{run_name}/stance_detection/{simple_data_name}_input.csv")

    logger.info("Printing the stats of stance detection dataset...")
    print_stats_of_stance_detection_df(stance_detection_df)

    model_names = [ "madlag/bert-large-uncased-mnli", 
                    "anirudh21/albert-large-v2-finetuned-mnli",
                    "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
                    "facebook/bart-large-mnli",
                    "roberta-large-mnli", 
                    "microsoft/deberta-large-mnli", 
                    ]
    
    for model_name in model_names:
        simple_model_name = model_name.split('/')[-1].split('-')[0]
        print(f"\n########################### {simple_model_name} ####################################")
        stance_prediction_output_path = f"../data/results/{run_name}/stance_detection/{simple_data_name}_{simple_model_name}.csv"

        stance_detector = StanceDetector(model_name, stance_detection_df, 
                                         batch_size=batch_size)
        stance_detector.output_path = stance_prediction_output_path
        evaluator = StanceDetectorEvaluator(stance_detector)

        evaluator.evaluate()
