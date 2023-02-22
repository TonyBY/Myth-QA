from typing import List, Optional, Dict
import logging

from pygaggle.qa.base import Answer, Question, Context
from pygaggle.qa.dpr_reader import DprReader
from .dpr_span_selection import UniqueDprSelection

class MultiAnswerDprReader(DprReader):
    def predict(
            self,
            question: Question,
            contexts: List[Context],
            topk_retrievals: Optional[List[int]] = None,
    ) -> Dict[int, List[Answer]]:
        """
        topk_retrievals: number of passages that are going to be used to answer the questions.
        """

        if self.span_selection_rules is None:
            self.span_selection_rules = [UniqueDprSelection()]

        if isinstance(question, str):
            question = Question(question)
        if topk_retrievals is None:
            topk_retrievals = [len(contexts)]

        logging.debug("===================================")
        logging.debug(f"Question: {question.text}")

        answers = {str(rule): {} for rule in self.span_selection_rules}
        prev_topk_retrieval = 0
        for rule in self.span_selection_rules:
            rule.reset()

        spans = self.compute_spans(question, contexts) # spans: List[List[span]]

        for topk_retrieval in topk_retrievals:
            logging.debug("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            logging.debug(f"topk_retrieval: {topk_retrieval}")
            for rule in self.span_selection_rules:
                rule.add_unique_answers(
                    spans[prev_topk_retrieval: topk_retrieval],
                    contexts[prev_topk_retrieval: topk_retrieval]
                )
                answers[str(rule)][topk_retrieval] = rule.top_answers(self.num_spans)

            prev_topk_retrieval = topk_retrieval

        # answers: {'rule': {'topk': [Answer]}} 
        return answers
