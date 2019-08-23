import spacy

NLP = spacy.load("en_core_web_sm")

def word_tokenize(utt):
    analyzed_utt = NLP(utt)
    return [token.lemma_ for token in analyzed_utt]

def lf_tokenize(lf):
    tokens = lf.replace("(", "( ").replace(")", " )").split()
    return tokens

def lf_detokenize(tokens):
    lf = " ".join(tokens).replace("( ", "(").replace(" )", ")")
    return lf
