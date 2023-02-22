from typing import List, Optional
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from Helper.utils import logger, set_device

class StanceDetector():
    """
    This class contains funtions to do stance detection using pretrained transformers.

    Paras:
    model_name: pretrained model name.
    data_df: DataFrame of the dataset for stance detection task.
    output_path: file path to save the stance detection predictions.
    tokenizer_name: pretrained tokenizer name.
    label2id: a map from label name to label id.
    id2label: a map from label id to label name.
    batch_size: number of claim pairs to pass in the model to process in each batch.
    device: computation hardware to use.
    """
    def __init__(
            self,
            model_name: str,
            data_df: Optional[pd.DataFrame]=None,
            output_path: Optional[str] = None,
            tokenizer_name: Optional[str] = None,
            label2id: dict = {"supporting": 1, "refuting": 0, "neutral": 2},
            id2label: dict = {0: "refuting", 1: "supporting", 2: "neutral"},
            batch_size: int = 32,
            device = None,
    ):
        self.data_df = data_df
        self.output_path = output_path

        self.label2id = label2id
        self.id2label = id2label

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        
        if device == None:
            self.set_device()
        else:
            self.device = device
        self.model = self.model.to(self.device).eval()

        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.batch_size = batch_size

    def set_device(self):
        logger.info(f"self.model.num_parameters(): {self.model.num_parameters()}")
        if self.model.num_parameters() < 10**9:
            self.device = set_device()
        else:
            logger.warning("Model is too large, putting it on cpu.")
            self.device =  torch.device("cpu")

    def tokenize_function(self, example: Dataset):
        return self.tokenizer(
            example["premise"], example["hypothesis_text"], padding="max_length", truncation=True, max_length=128
        )

    def get_tokenized_dataset_from_df(self) -> Dataset:
        raw_dataset = Dataset.from_pandas(self.data_df)
        return raw_dataset.map(self.tokenize_function, batched=True)

    def get_eval_dataloader(self):
        data_collator = DataCollatorWithPadding(self.tokenizer)

        tokenized_datasets = self.get_tokenized_dataset_from_df()
        logger.debug(f"tokenized_datasets.column_names: {tokenized_datasets.column_names}")
        try:
            tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis_ids", "hypothesis_text", "stance", "pred_label_ids", "pred_label_text"])
        except:
            try:
                tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis_ids", "hypothesis_text", "stance"])
            except:
                tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis_text"])

        return DataLoader(tokenized_datasets, 
                                     batch_size=self.batch_size, 
                                     collate_fn=data_collator)

    def do_stance_detection(self) -> List[int]:
        eval_dataloader = self.get_eval_dataloader()
        predictions, _ = self.get_model_predictions(eval_dataloader)
        self.data_df["pred_label_ids"] = predictions
        self.data_df["pred_label_text"] = self.data_df.pred_label_ids.map(self.model.config.id2label)
        if self.output_path != None:
            logger.info(f"Saving the predictions to {self.output_path}")
            self.save_predictions()
        return self.data_df

    def save_predictions(self):
        logger.info(f"Stance detection predictions are saved to: {self.output_path}")
        self.data_df.to_csv(self.output_path)

    def get_model_predictions(self, eval_dataloader):
        predictions = []
        scores = []
        # for batch in tqdm(eval_dataloader, desc="Doing stance detection..."):
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)
            predictions.extend(pred.cpu().detach().numpy())
            score = torch.max(logits, dim=-1)
            scores.extend(score.values.cpu().detach().numpy())
        return predictions, scores
