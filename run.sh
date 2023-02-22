#!/bin/bash
date;hostname;pwd

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

export PYTHONPATH=./MythQA

DATASET_PATH="./data/annotations/TweetMythQA.jsonl"
PROCESSED_CORPUS_DIR="./data/processed_tweets/"
RESULT_DIR="./data/results/end2end/"
LOG_PATH="./data/results/end2end/evals/$(date).txt"

BM25_INDEX_PATH="./data/index/sparse_term_frequency_embedding"
DPR_INDEX_PATH="./data/index/dindex-sample-dpr-multi"

BATCH_SIZE=32
NUMBER_OF_RETRIEVALS=1000
k1=1.6
b=0.75
ANSWER_NUMBER=10

RETRIEVER="bm25" # or "dpr"
QUERY_ENCODER_NAME="facebook/dpr-question_encoder-multiset-base"
READER="dpr_reader" # or "t5"
T5_READER_MODEL_NAME="valhalla/t5-base-qa-qg-hl"
DPR_READER_MODEL_NAME="facebook/dpr-reader-single-nq-base"
STANCE_DETECTOR_MODEL_NAME="microsoft/deberta-large-mnli"
DPR_SETTINGS="dprfusion_1.0_0.55"

T1=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python main.py --dataset "$DATASET_PATH" --cache-results --use-cache --index-path "$BM25_INDEX_PATH" --number-of-retrievals "$NUMBER_OF_RETRIEVALS" --num-asnwers "$ANSWER_NUMBER" --pretrained-t5-reader-model "$T5_READER_MODEL_NAME" --pretrained-dpr-reader-model "$DPR_READER_MODEL_NAME" --dpr-settings "$DPR_SETTINGS" --pretrained-nli-model "$STANCE_DETECTOR_MODEL_NAME" > "$LOG_PATH"
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
