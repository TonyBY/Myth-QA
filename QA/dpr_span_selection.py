import logging
from pygaggle.qa.base import Answer
from pygaggle.qa.span_selection import DprSelection
from pygaggle.model.evaluate import ReaderEvaluator

class UniqueDprSelection(DprSelection):
    def reset(self):
        self.answers = []
        self.viewed_span_text = set()
           
    def add_unique_answers(self, spans_by_text, texts):
        for spans, text in zip(spans_by_text, texts):
            logging.debug("####################################")
            logging.debug("Context: ", text.text.lower().strip("""!"#&\'()*+,-./:;<=>?@[\\]^_`{|}~ """))
            context_word_set = set(text.text.lower().split())
            for span in spans:
                if not set(span.text.strip().split()).issubset(context_word_set):
                    if ("").join(span.text.strip().split()) not in context_word_set:
                        logging.debug("span.text: ", span.text)
                        continue
                    else:
                        normalized_span_text = ("").join(span.text.strip().split())
                else:
                    normalized_span_text = ReaderEvaluator._normalize_answer(span.text)
                if normalized_span_text in self.viewed_span_text:
                    continue
                self.viewed_span_text.add(normalized_span_text)
                self.answers.append(Answer(text=normalized_span_text,
                                           context=text,
                                           score=self.score(span, text)))


class UniqueDprFusionSelection(UniqueDprSelection):
    def __init__(self, beta, gamma):
        self.beta = float(beta)
        self.gamma = float(gamma)

    def score(self, span, text):
        return float(span.relevance_score) * self.beta + float(text.score) * self.gamma, float(span.span_score)

    def __str__(self):
        return f'DPR Fusion, beta={self.beta}, gamma={self.gamma}'


def parse_span_selection_rules(settings):
    settings = settings.split('_')

    settings_map = dict(
        dpr=UniqueDprSelection,
        dprfusion=UniqueDprFusionSelection
    )
    return settings_map[settings[0]](*settings[1:])
