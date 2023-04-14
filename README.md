# MythQA: Query-based Large-scale Check-Worthy Claim Detection through Multi-Answer Open-domain Question Answering

Yang Bai, Anthony Colas, Daisy Wang<br>
University of Florida <br>

## Getting Started
We describe here how to run the pipeline end-to-en and how to evaluate each module seperatly. 
Before started, make sure you get your environment ready.<br>
- Python >= 3.8<br>
- Java >= 15.0<br>
```bash
pip install -r requirements.txt
```
- Add this project directory to your system environment.<br>
```bash
export PYTHONPATH=$<MythQA project dir>
```

## Data Preparing
- Annotated dataset is saved at 'data/annotations/' directory.<br>
- Tweet id corpus is saved at 'data/id_corpus/' directory.<br>
- [Collect raw tweet corpus through TwitterAPI](#getting-tweet-content-using-tweet-ids-through-twitter-api)
- [Generate cleaned textual corpus from raw tweets](#getting-precessed-tweets-from-raw)
- [Generate indexed corpus](#generating-indexed-corpus-from-textual-corpus)

### Getting tweet content using tweet ids through Twitter API
- cd into the 'DataProcess/' directory.
```bash
python TwitterAPIQuerier.py
```
- Returned raw tweets will be saved at 'data/raw/' directory.
- Note: Twitter API access is required to get tweet content with tweet ids.
- By the [Twitter's policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy#id34:~:text=The%20best%20place,commercial%20research%20purposes.), we are not allowed to share tweet content but tweet IDs.

### Getting precessed tweets from raw
- cd into the 'DataProcess/' directory.
```bash
python tweet_process.py 
```
- Returned processed tweets will be saved at 'data/processed_tweets/' directory.<br>
- This step is necessary for downstream operations.

### Generating indexed corpus from textual corpus
- cd into the 'DataProcess/' directory.
```bash
python index_corpus.py
```
- Dense index will be saved at data/index/dindex-sample-dpr-multi directory.<br>
- Sparse index will be saved at data/index/sparse_term_frequency_embedding directory.<br>
- This step is necessary for downstream operations

## End-to-end demo
- At the root directory.
```bash
python main.py --demo 'What is origin of COVID-19?' --cache-results --use-cache
```
- Result will be saved at 'data/resutls/end2end/demos/' directory.

## Evaluate Tweet Retrieval
- cd into the 'Evaluator/' directory.
```bash
python retriever_evaluator.py
```

## Evaluate Multiple Answer Prediction for Entity Questions.
### Intrinsic Evaluation
- cd into the 'Evaluator/' directory.
```bash
python reader_evaluator.py --intrinsic
```

### Extrinsic Evaluation
- cd into the 'Evaluator/' directory.
```bash
python reader_evaluator.py
```

## Evaluate Stance Detection.
- cd into the 'Evaluator/' directory.
```bash
python stance_detector_evaluator.py
```

## Evaluate Controversail Stance Mining.
### Intrinsic Evaluation
- cd into the 'Evaluator/' directory.
```bash
python stance_miner_evaluator.py --intrinsic
```

### Extrinsic Evaluation
- cd into the 'Evaluator/' directory.
```bash
python stance_miner_evaluator.py
```

- All outputs can be found in the './data/results/' directory.

## Metrics
- New metrics are proposed for evaluation: MHit@k for multi-answer IR; F1_ans, F1_CONTRO@e for end-to-end MythQA evaluation. Please refer to our paper for more details.

## Citation
If you use MythQA, please cite the following paper: 

```
Tobe added
```

## Acknowledgments
- [Pyserini library](https://github.com/castorini/pyserini) for efficient informaton retrieval implementations.<br>
- Our DPR reader code is borrowed from [PyGaggle](https://github.com/castorini/pygaggle).<br>
- All the pre-trained models are acquired from the [Huggingface.co](https://huggingface.co/).<br>

We thank all the authors for their useful code. 
