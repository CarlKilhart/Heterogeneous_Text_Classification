from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class tfidf_buf:
    def __init__(self, vocab_set) -> None:
        self.fit_flag = False
        self.vocab_set = vocab_set
        self.count_tf = CountVectorizer(vocabulary=self.vocab_set)
        self.tfidf_tf = TfidfTransformer(norm=None)
        # 生成单词映射
        self.word2id = dict()
        for v, i in enumerate(self.vocab_set):
            self.word2id[i] = v

    def fit(self, text):
        self.fit_flag = True
        count_vec = self.count_tf.fit_transform(text)
        tfidf_vec = self.tfidf_tf.fit_transform(count_vec)
        
        return tfidf_vec
    
    def transform(self, text):
        if not self.fit_flag:
            raise RuntimeError('Please fit first!')
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise RuntimeError('text is not str or list of str')
        
        count_vec = self.count_tf.transform(text)
        tfidf_count = self.tfidf_tf.transform(count_vec)
        
        return tfidf_count