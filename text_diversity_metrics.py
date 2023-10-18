import pandas as pd
from textdiversity import (
    TokenSemantics, DocumentSemantics, AMR, # semantics
    DependencyParse, ConstituencyParse,     # syntactical
    PartOfSpeechSequence,                   # morphological
    Rhythmic                                # phonological
)
from lexical_diversity import lex_div as ld
from nltk import ngrams
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import types
import argparse


class LDHelper:

    def _flemmatize(self, corpus):
        flemmas = []
        for doc in corpus:
            flemmas.extend(ld.flemmatize(doc))
        return flemmas

    def ttr(self, coprus):
        return ld.ttr(self._flemmatize(coprus))

    def root_ttr(self, coprus):
        return ld.root_ttr(self._flemmatize(coprus))

    def log_ttr(self, coprus):
        return ld.log_ttr(self._flemmatize(coprus))

    def maas_ttr(self, coprus):
        return ld.maas_ttr(self._flemmatize(coprus))

    def msttr(self, coprus):
        return ld.msttr(self._flemmatize(coprus))

    def mattr(self, coprus):
        return ld.mattr(self._flemmatize(coprus))

    def hdd(self, coprus):
        return ld.hdd(self._flemmatize(coprus))

    def mtld(self, coprus):
        return ld.mtld(self._flemmatize(coprus))

    def mtld_ma_wrap(self, coprus):
        return ld.mtld_ma_wrap(self._flemmatize(coprus))

    def mtld_ma_bid(self, coprus):
        return ld.mtld_ma_bid(self._flemmatize(coprus))


class UniqueNgramHelper:

    def _tokenize(self, corpus):
        tokens = []
        for doc in corpus:
            tokens.extend(word_tokenize(doc))
        return tokens

    def _make_unique(self, n_gram_generator):
        return len(set(list(n_gram_generator)))

    def unigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 1)
        return self._make_unique(n_gram_generator)

    def bigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 2)
        return self._make_unique(n_gram_generator)

    def trigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 3)
        return self._make_unique(n_gram_generator)
    
def get_metrics(dataset_path, output_path, column_name, metrics):
        
    dataset = load_dataset(dataset_path)
    prompts = [s for s in dataset[column_name] if s.strip() != "" and type(s) == str]

    config = {"normalize": False}

    ldhelper = LDHelper()
    unhelper = UniqueNgramHelper()

    metric_funcs = {
        'TokenSemantics': TokenSemantics(config), 
        'DocumentSemantics': DocumentSemantics(), # AMR(config),
        'DependencyParse': DependencyParse(config), 
        'ConstituencyParse': ConstituencyParse(config),
        'PartOfSpeechSequence': PartOfSpeechSequence(config),
        'Rhythmic': Rhythmic(config),
        'ttr': ldhelper.ttr,
        'log_ttr': ldhelper.log_ttr,
        'root_ttr': ldhelper.root_ttr,
        'maas_ttr': ldhelper.maas_ttr,
        'mattr': ldhelper.mattr,
        'msttr': ldhelper.msttr,
        'hdd': ldhelper.hdd,
        'mtld': ldhelper.mtld,
        'mtld_ma_bid': ldhelper.mtld_ma_bid,
        'mtld_ma_wrap': ldhelper.mtld_ma_wrap,
        'unigrams': unhelper.unigrams,
        'bigrams': unhelper.bigrams,
        'trigrams': unhelper.trigrams,
    }

    results = []
    for metric_name in metrics:
        if metric_name in metric_funcs:
            results.append({
            "metric": metric_name,
            "corpus": metric_funcs[metric_name](prompts)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Diversity Metrics Args')
    parser.add_argument("-prompts_path", required=True)
    parser.add_argument("-metrics", required=True)
    parser.add_argument("-column_name", default='prompts')
    parser.add_argument("-output_path", default='metrics.csv')
    args = parser.parse_args()
    get_metrics(args.prompts_path, args.output_path, args.column_name, args.metrics)