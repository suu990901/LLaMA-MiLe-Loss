"""
Natural Questions: a Benchmark for Question Answering Research
https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf

The Natural Questions (NQ) corpus is a question-answering dataset that contains
questions from real users and requires QA systems to read and comprehend an entire
Wikipedia article that may or may not contain the answer to the question. The
inclusion of real user questions, and the requirement that solutions should read
an entire page to find the answer, cause NQ to be a more realistic and challenging
task than prior QA datasets.

TODO: NaturalQS has a *really* large train set that huggingface just automatically
downloads even if you dont use it. we should try and only download the val set and
not even bother with the train set.

Homepage: https://ai.google.com/research/NaturalQuestions
"""
import json
from lm_eval.base import Task, GenerateTask
import re
import string
_CITATION = """
@article{47761,
    title={Natural Questions: a Benchmark for Question Answering Research},
    author={Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
    year={2019},
    journal={Transactions of the Association of Computational Linguistics}
}
"""


#移除全部的在string.punctuation中的标点符号
def replace_all(text):
    for punctuation in string.punctuation:
        text=text.replace(punctuation, '')
    return text

#移除全部的冠词
def remove_articles(text):
    #冠词的列表
    articles=['an ','a ','the ','some ']
    for article in articles:
        if text.startswith(article):
            text=text[len(article):]
            break
    return text

def normalize(text):
    text=text.strip().lower()
    #移除所有符号
    text=replace_all(text).strip()
    #移除冠词
    # text=remove_articles(text).strip()
    #移除连续空格
    text=re.sub('\s+',' ',text)
    return text


class NaturalQs(GenerateTask):
    VERSION = 0
    DATASET_PATH = "natural_questions"
    DATASET_NAME = None
    STOP_FLAGS = [60,'\n']
    Save=[]
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # print('downloading')
        data=[]
        with open('/nlp_group/zhangyusong/eval_llm/datasets/natural-questions/nq_open/NQ-open.dev.jsonl','r') as f:
            val_data=[json.loads(line) for line in f]
        with open('/nlp_group/zhangyusong/eval_llm/datasets/natural-questions/nq_open/NQ-open.train.jsonl','r') as f:
            train_data=[json.loads(line) for line in f]
        assert len(val_data)==3610
        assert len(train_data)==87925
        #构建
        val_prompts=[]
        for idx in range(len(val_data)):
            temp={
                "id":idx,
                "raw_question":val_data[idx]['question'],
                "label": val_data[idx]['answer']
            }
            val_prompts.append(temp)
        train_prompts=[]
        for idx in range(len(train_data)):
            temp={
                "id":idx,
                "raw_question":train_data[idx]['question'],
                "label": train_data[idx]['answer']
            }
            train_prompts.append(temp)
        self.dataset = {"validation": val_prompts, "train": train_prompts}

    #用于生成Q:的部分
    def doc_to_text(self, doc):
        return "Q: " + doc["raw_question"] +"?"
    #few-shot,用于生成A:的部分
    def doc_to_target(self, doc):
        return "A: " + doc["label"][0]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False
    
    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]


    def fewshot_context(
        self, doc, num_fewshot, rnd=None, description="Answer these questions:"
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n".join(
                    [
                        self.doc_to_text(doc) +'\n' + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n"
            )

        example = self.doc_to_text(doc)+"\nA:"
        return description + labeled_examples + example

    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        def compare(model_output,ground_truths):
            if model_output in ground_truths:
                return 1
            else:
                return 0
        model_output = normalize(results[0])
        ground_truths=list(map(normalize,doc["label"]))
        exactly_match = compare(model_output, ground_truths)
        doc['response'] = results[0]
        doc['answer'] = doc["label"]
        doc['extracted'] = model_output
        doc['score'] = str(exactly_match)

        return {'acc': exactly_match}


if __name__=="__main__":
    task=NaturalQs()
    task.download()
