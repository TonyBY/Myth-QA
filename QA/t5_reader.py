from typing import List
from pygaggle.qa.base import Reader, Answer, Question, Context
from pygaggle.model.evaluate import ReaderEvaluator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from Helper.utils import logger, set_device


class T5Reader(Reader):
    """Class containing the T5 Reader
    Takes in a question and a list of the top passages selected by the retrieval model,
    and predicts a list of the best answer spans from the most relevant passages.

    Parameters
    ----------
    model_name : T5 Reader model name or path
    tokenizer_name : T5 Reader tokenizer name or path
    num_answers : Number of answers to return
    """
    
    def __init__(
            self,
            model_name: str,
            tokenizer_name: str = None,
            num_answers: int = 1,
            batch_size: int = 16,
            device = None
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == None:
            self.set_device()
        else:
            self.device = device
        self.model = self.model.to(self.device).eval()
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_answers = num_answers
        self.batch_size = batch_size
        self.span_selection_rules = ['t5']
        
    def set_device(self):
        if self.model.num_parameters() < 10**9:
            self.device =  set_device()
        else:
            logger.warning("Model is too large, putting it on cpu.")
            self.device =  torch.device("cpu")
            
    def predict(self, question: Question, contexts: List[Context], topk_retrievals=[100]) -> List[List[Answer]]:
        answer_objs = []
        answers = {span_selection_rule: {} for span_selection_rule in self.span_selection_rules}
        logger.debug(f"len(contexts): {len(contexts)}")
        for i in range(0, len(contexts), self.batch_size):
            logger.debug(f"i: {i}")
            input_sequences = []
            if i + self.batch_size > len(contexts):
                b_range = len(contexts) - i
            else:
                b_range = self.batch_size
            for b in range(b_range):
                logger.debug(f"b: {b}")
                logger.debug(f"i+b: {i+b}")
                input_sequence = 'question: ' + question.text + ' context: ' + contexts[i+b].text +'</s>'
                input_sequences.append(input_sequence)
                logger.debug(f"input_sequences: {input_sequences}")

            encoded_inputs = self.tokenizer(input_sequences,
                                            padding="longest",
                                            max_length=512,
                                            truncation=True,
                                            return_tensors="pt")

            logger.debug(f"encoded_inputs: {encoded_inputs}")
            input_ids  = encoded_inputs.input_ids.to(self.device)
            logger.debug(f"input_ids: {input_ids}")

            gen_output = self.model.generate(input_ids)
            logger.debug(f"gen_output: {gen_output}")

            predicted_answers = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)
            logger.debug(f"predicted_answers: {predicted_answers}")

            for idx, pa in enumerate(predicted_answers):
                answer_objs.append(Answer(text=ReaderEvaluator._normalize_answer(pa), 
                                            context=contexts[i+idx], 
                                            score=contexts[i+idx].score))

        logger.debug(f"len(answer_objs): {len(answer_objs)}")
        for topk_retrieval in topk_retrievals:
            logger.debug(f"topk_retrieval: {topk_retrieval}")
            viewed_asnwers = set()
            for span_selection_rule in self.span_selection_rules:
                answers[span_selection_rule][topk_retrieval] = []
                for t in range(topk_retrieval):
                    logger.debug(f"t: {t}")
                    logger.debug(f'len(answers[span_selection_rule][topk_retrieval]): {len(answers[span_selection_rule][topk_retrieval])}')
                    logger.debug(f'self.num_answers: {self.num_answers}')
                    if len(answers[span_selection_rule][topk_retrieval]) >= self.num_answers or t >= len(answer_objs):
                        break
                    if answer_objs[t].text not in viewed_asnwers and len(answer_objs[t].text.split(' ')) <= 5:
                        answers[span_selection_rule][topk_retrieval].append(answer_objs[t])
                        viewed_asnwers.add(answer_objs[t].text)
        return answers
