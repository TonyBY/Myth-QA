# MythQA: Query-based Large-scale Check-Worthy Claim Detection through Multi-Answer Open-domain Question Answering

Yang Bai, Anthony Colas, Daisy Wang<br>
University of Florida <br>

## 0. Getting Started
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

## 1. Data Preparing
- Annotated dataset is saved at 'data/annotations/' directory.<br>
  - An Example of Data
```jsonl
[
  {"question": "What can spread COVID-19?", 
   "question_type": "entity", 
   "topic": "COVID-19 medium", 
   "answers": {
        "shoes": {
              "supporting": ["1245463329084235778", "1249874483562700800", "1242423394953560064", "1251440294173634560", "1249835506927534082", "1251239264156860421", "1243957639689592833", "1245208383239540736", "1246885147741478913", "1251252879928643584"], 
              "refuting": ["1262297385578967040", "1451520936658685953", "1278938775221964800", "1242366875281035265"], 
              "neutral": []}, 
        "swimming water": {
              "supporting": ["1403105745311125504"], 
              "refuting": ["1263008104934973440", "1261110821939281920", "1332337910264201216", "1261273142376271873", "1300440198229168128", "1260980292669456384", "1288472810843541507", "1266406475246833664", "1261176857996800000", "1262941118691454978"], 
              "neutral": []}, 
        "pets": {
              "supporting": ["1384991584173842437", "1377408233531338753", "1377190516882866177", "1377526917541289985", "1376509873383411714", "1415851746186825732"], 
              "refuting": ["1267835896650489858", "1376211259826319361", "1311371486339518464", "1277952295507054595", "1361727923997310982", "1420624531631132678"], 
              "neutral": []}, 
        "cats": {
              "supporting": ["1294238023287111682", "1398675435895017476", "1311856579919122434", "1376942130799767555", "1356852593985679362", "1342464451161702400", "1261348646353764355", "1304855698438848512"], 
              "refuting": ["1267835896650489858", "1373357182897561602", "1261758187880894464", "1261887702217027584", "1270037402782965760", "1287819054925873153", "1280409353686781952", "1440774935102705669", "1420791212471197702", "1262156207118340098", "1397048459862020096", "1460693240336855047"], 
              "neutral": []}, 
        "dogs": {
              "supporting": ["1459571452844277765", "1429896369137934336", "1377251191843811329", "1429795567136350208", "1353743026842177536", "1429456592995594253", "1376468124829122562", "1353950144853663746", "1473661506478821382", "1448671489268412427"], 
              "refuting": ["1362047785189707782", "1448567432327618561", "1373357182897561602", "1337528145721876480", "1311856579919122434", "1350137189305262080", "1463520062925688832", "1395808389473705985", "1379792411652550663", "1458065247500226560", "1460693240336855047", "1246179788978556929"], 
              "neutral": []}, 
        "farm animals": {
              "supporting": ["1379855134549700609", "1297252256212033537", "1377211921439543298", "1377553895103283202", "1314688905673834506", "1300411291182477312", "1377218819584851972", "1362965146340126720"], 
              "refuting": ["1336102685230161921", "1384991584173842437"], 
              "neutral": []}, 
        "minks": {
              "supporting": ["1324517600932601858", "1332936725417508864", "1340008192340848640", "1324849754241720321", "1325704356369489922", "1315049707958403073", "1331874464208809984", "1324053058138083330", "1324330588338233345", "1324941333384015874", "1328811175178592257", "1227285253988831235", "1328431057738362880", "1346177592785821701", "1332134545630629892"], 
              "refuting": ["1325140517068238848", "1312542202803953666"], 
              "neutral": []}, 
        "cows": {
              "supporting": ["1346862469835337729", "1448728842122706945", "1391246188179902469"], 
              "refuting": ["1336102685230161921", "1284741321337577472", "1384991584173842437"], 
              "neutral": []}, 
        "chicken": {
              "supporting": ["1297872292425408512", "1243500947017433089", "1293912407614472193"], 
              "refuting": ["1236501898833940481", "1239939558554234880", "1245353245914079237", "1236129924575985664", "1384991584173842437", "1336102685230161921", "1312769586895499265", "1236633078216265728", "1247245273018945537", "1239673276252618752", "1246179788978556929"], 
              "neutral": []}, 
        "pigs": {
              "supporting": ["1229269920237658112", "1251045659370717184", "1435175924866166786", "1245632495380385792", "1229839739735375873", "1243500947017433089", "1243214563350384646", "1315447419870474240"], 
              "refuting": ["1245353245914079237", "1384991584173842437", "1336102685230161921", "1312769586895499265", "1246179788978556929"], 
              "neutral": []}, 
        "bats": {
              "supporting": ["1275046393414025218", "1250135439098277888", "1227285253988831235", "1245707930579742720", "1435175924866166786", "1267802317270716417", "1262385100773261312", "1245632495380385792", "1242238344874033152", "1461065223591415808", "1354073432040865797"], 
              "refuting": ["1242687863952736256", "1376930373280669699", "1261288874250297347", "1238202833184440334", "1254367487266713600"], 
              "neutral": []}, 
        "pangolins": {
              "supporting": ["1253810807378456582", "1275046393414025218", "1226071282858352640", "1227285253988831235", "1348200906270126084", "1226203318076289030", "1245707930579742720", "1238202833184440334", "1354073432040865797", "1242238344874033152", "1237248185548058624", "1256655022089068544"], 
              "refuting": ["1226404770849538049", "1349263876903366656", "1231347983427526658", "1226395823526809600"], 
              "neutral": []}, 
        "mosquitoes": {
              "supporting": ["1295160476192501760"], 
              "refuting": ["1248744264617447424", "1285097379771842560", "1240593269949677568", "1298327532824035330", "1250807454847262720", "1285983451166978050", "1261278995477299200", "1264264122788646913", "1285930118582538244"], 
              "neutral": []}, 
        "air": {
              "supporting": ["1467167385702608900", "1459992986105049088", "1464631504844341255", "1470868579633541133", "1468268064076492801", "1465598639859015682", "1465365461105876998"], 
              "refuting": ["1473004530161917962"], 
              "neutral": []}, 
        "swimming pools": {
              "supporting": ["1241730831317680130", "1242526583321006080", "1372988000385167361", "1240173596498067458", "1272556475823636482"], 
              "refuting": ["1260329193080053760", "1379652773327761413", "1288893193237467139", "1281482650226692097", "1410231752438734853", "1413602358907506695", "1277061686424297473"], 
              "neutral": []}, 
        "hot tubs": {
              "supporting": ["1266437730940174336", "1287545661773058048"], 
              "refuting": ["1276235764364345351", "1267173797368598529", "1379652773327761413", "1262965555088121856", "1261666212804571137", "1281482650226692097", "1413602358907506695"], 
              "neutral": []}}}

]
```
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

## 2. End-to-end demo
- At the root directory.
```bash
python main.py --demo 'What is origin of COVID-19?' --cache-results --use-cache
```
- Result will be saved at 'data/resutls/end2end/demos/' directory.

## 3. Evaluate Tweet Retrieval
- cd into the 'Evaluator/' directory.
```bash
python retriever_evaluator.py
```

## 4. Evaluate Multiple Answer Prediction for Entity Questions.
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

## 5. Evaluate Stance Detection.
- cd into the 'Evaluator/' directory.
```bash
python stance_detector_evaluator.py
```

## 6. Evaluate Controversail Stance Mining.
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

## 7. Metrics
- New metrics are proposed for evaluation: MHit@k for multi-answer IR; F1_ans, F1_CONTRO@e for end-to-end MythQA evaluation. Please refer to our paper for more details.

## Cite
If you use MythQA, please cite the following paper: 

```
@inproceedings{10.1145/3539618.3591907,
author = {Bai, Yang and Colas, Anthony and Wang, Daisy Zhe},
title = {MythQA: Query-Based Large-Scale Check-Worthy Claim Detection through Multi-Answer Open-Domain Question Answering},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591907},
doi = {10.1145/3539618.3591907},
abstract = {Check-worthy claim detection aims at providing plausible misinformation to the downstream fact-checking systems or human experts to check. This is a crucial step toward accelerating the fact-checking process. Many efforts have been put into how to identify check-worthy claims from a small scale of pre-collected claims, but how to efficiently detect check-worthy claims directly from a large-scale information source, such as Twitter, remains underexplored. To fill this gap, we introduce MythQA, a new multi-answer open-domain question answering(QA) task that involves contradictory stance mining for query-based large-scale check-worthy claim detection. The idea behind this is that contradictory claims are a strong indicator of misinformation that merits scrutiny by the appropriate authorities. To study this task, we construct TweetMythQA, an evaluation dataset containing 522 factoid multi-answer questions based on controversial topics. Each question is annotated with multiple answers. Moreover, we collect relevant tweets for each distinct answer, then classify them into three categories: "Supporting", "Refuting", and "Neutral". In total, we annotated 5.3K tweets. Contradictory evidence is collected for all answers in the dataset. Finally, we present a baseline system for MythQA and evaluate existing NLP models for each system component using the TweetMythQA dataset. We provide initial benchmarks and identify key challenges for future models to improve upon. Code and data are available at: https://github.com/TonyBY/Myth-QA},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {3017–3026},
numpages = {10},
keywords = {check-worthy claim detection, multi-answer open-domain question answering, social media, natural language inference},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```

## Acknowledgments
- [Pyserini library](https://github.com/castorini/pyserini) for efficient informaton retrieval implementations.<br>
- Our DPR reader code is borrowed from [PyGaggle](https://github.com/castorini/pygaggle).<br>
- All the pre-trained models are acquired from the [Huggingface.co](https://huggingface.co/).<br>

We thank all the authors for their useful code. 
