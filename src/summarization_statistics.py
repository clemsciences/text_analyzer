# -*- coding: utf8 -*-

import re
import math

import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict

__author__ = "besnier"

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
stoplist = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.SnowballStemmer('english')


class NotCalculatedExceptionYet(ValueError):
    """
    It's when we want ti get an attribute but that was not computed yet
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class SentenceFeatures:
    """
    Compute the sentence features defined in [Chen et al. 2015]
    """
    def __init__(self, documents, old_document, new_documents):
        self.old_documents = old_document
        self.new_documents = new_documents
        self.documents = documents
        self.res = []
        self.features = {"f11": None, "f12": None, "f13": None, "f14": None, "f15": None, "f16": None, "f17": None,
                         "f18": None, "f19": None}

        self.alpha = 0.1  # parameter used to give more weight either to unigrams or bigrams

    def update_feature_extraction_sentence(self, sent, sentence_position,
                                           number_sentence_in_doc, summary_from_old_data=""):
        """
        Values are normalized in order to be comparable between live blogs
        :param sent:
        :param sentence_unigrams: string of a sentence
        :param sentence_bigrams: list of sentences
        :param sentence_position
        :param number_sentence_in_doc
        :param summary_from_old_data
        :return:
        """
        if len(sent.list_of_unigrams)+len(sent.list_of_bigrams) != 0:
            # F11 - interpolated n-gram document frequency
            self.features["f11"] = (self.alpha * sum([self.new_documents.document_frequency_for_unigrams[unigram]
                                                     for unigram in sent.list_of_processed_unigrams]) +
                                   (1 - self.alpha) * sum([self.new_documents.document_frequency_for_bigrams[bigram]
                                                           for bigram in sent.list_of_processed_bigrams])) / \
                                   (len(sent.list_of_unigrams)+len(sent.list_of_bigrams))
        else:
            self.features["f11"] = 0
        # F12 - sentence position in that position
        self.features["f12"] = sent.position_in_the_document/len(self.documents.list_of_documents[sent.document_id].list_of_sentences)
        # F13 - is the sentence in the first 1/2/3 position in that document?
        self.features["f13"] = delta(sentence_position in [0, 1, 2])
        # F14 - is the sentence in the last 1/2/3 position in that document?
        self.features["f14"] = delta(sentence_position in [number_sentence_in_doc - 3, number_sentence_in_doc - 2,
                                                           number_sentence_in_doc - 1])
        # F15 - sentence length
        self.features["f15"] = len(sent.list_of_unigrams)  # number of tokens in the sentence
        # f_16 = sentence similarity with topic's query not implemented because there is no query here

        # f_17  # It needs the summary of old data
        self.features["f17"] = sent.cos_tfidf(summary_from_old_data, self.documents)

        # F18 - interpolated n-gram novelty
        if len(sent.list_of_unigrams)+len(sent.list_of_bigrams) != 0:
            self.features["f18"] = (self.alpha * sum(
                [self.new_documents.document_frequency_for_unigrams[unigram]/
                 float(self.old_documents.document_frequency_for_unigrams[unigram] +
                       len(self.documents.list_of_documents)) for unigram in sent.list_of_processed_unigrams]) +
                        (1 - self.alpha) * sum(
                                       [self.new_documents.document_frequency_for_bigrams[bigram] /
                                        float(self.old_documents.document_frequency_for_bigrams[bigram] +
                                              len(self.documents.list_of_documents)) for bigram in
                                        sent.list_of_processed_bigrams])) / \
                                        (len(sent.list_of_unigrams)+len(sent.list_of_bigrams))

            # F19 - interpolated n-gram uniqueness
            self.features["f19"] = (self.alpha *
                sum([self.new_documents.document_frequency_for_unigrams[unigram] /
                     float(len(self.documents.list_of_documents)) for unigram in
                     sent.list_of_processed_unigrams
                     if self.old_documents.document_frequency_for_unigrams[unigram] > 0]) +
                (1 - self.alpha) *
                sum([self.new_documents.document_frequency_for_bigrams[bigram] /
                    float(len(self.documents.list_of_documents)) for bigram in
                    sent.list_of_processed_bigrams
                    if self.old_documents.document_frequency_for_bigrams[bigram] > 0])) / \
                                   (len(sent.list_of_unigrams)+len(sent.list_of_bigrams))
        else:
            self.features["f18"] = 0
            self.features["f19"] = 0

    def __str__(self):
        return str(self.features)

    def get_vectorized_features(self):
        return np.array([self.features["f11"], self.features["f12"], self.features["f13"], self.features["f14"],
                         self.features["f15"], self.features["f17"], self.features["f18"], self.features["f19"]])


class BigramFeatures:
    """
    easy features: f1, f5, f6, f7, f9, f10
    easy features: f11, f12, f13, f14, f15, f18, f19
    """

    def __init__(self, bigram, update_documents, pos_in_sentence, number_of_bigrams_in_sentence):
        self.res = []  # list of list of [word1, word2, dictionary of features]
        self.bigram_frequency_in_relevant_sentences = update_documents.bigram_frequency_in_relevant_sentences
        self.old_documents = update_documents.old_documents
        self.new_documents = update_documents.new_documents
        self.documents = update_documents.documents
        self.features = {"f1": 0, "f2": 0, "f3": 0, "f4": None, "f5": None, "f6": None, "f7": 0, "f8": 0, "f9": 0,
                         "f10": None}
        # self.unigram_features = defaultdict(lambda: {"old_documents": 0, "new_documents": 0})

        # F1 - Document frequency in new data set
        self.features["f1"] = self.new_documents.document_frequency_for_bigrams[bigram] / \
                              float(len(self.new_documents.list_of_documents))

        # F2 - normalized term frequency in all filtered relevant sentences
        self.features["f2"] = self.bigram_frequency_in_relevant_sentences[bigram]
        # F3 - sentence frequency in all relevant sentences
        self.features["f3"] = 0  # Strange

        # F4 - do bigram words appean in topic's query (is not here because we don't have a query)

        # F5 - is the bigram in the first 1/2/3 position of that sentence
        self.features["f5"] = pos_in_sentence in range(2)
        # F6 - is the bigram in the mast 1/2/3 position of that sentence
        self.features["f6"] = pos_in_sentence in [number_of_bigrams_in_sentence - 3,
                                                  number_of_bigrams_in_sentence - 2,
                                                  number_of_bigrams_in_sentence - 1]

        # F7 - document frequency in old data set -
        self.features["f7"] = self.old_documents.document_frequency_for_bigrams[bigram] / float(
            len(self.old_documents.list_of_documents))

        # F8 - normalized bigram frequency in the summary from old data set -
        self.features["f8"] = None  # We should have here the summary of

        # F9 - bigram novelty value
        self.features["f9"] = self.features["f1"] / \
                              float(self.features["f7"] + len(self.documents.list_of_documents))
        # F10 - bigram uniqueness value
        if self.features["f7"] > 0:
            self.features["f10"] = 0
        else:
            self.features["f10"] = self.features["f1"] / float(len(self.documents.list_of_documents))

    def __str__(self):
        return str(self.features)

    def get_vectorized_features(self):
        return np.array([self.features["f1"], self.features["f2"], self.features["f3"], self.features["f5"], self.features["f6"], self.features["f7"], self.features["f8"], self.features["f9"], self.features["f10"]])


class Sentence:
    def __init__(self):
        """

        :return:
        """
        self.raw_text = ""  # the raw form of the text without any processing

        self.list_of_unigrams = []  # list of all the tokens in the sentence
        self.list_of_bigrams = []  # list of the bigrams of the tokens in the sentence
        self.list_of_processed_bigrams = []  # list of bigrams which contain no stopwords and which are stemmed
        self.list_of_processed_unigrams = []  # list of unigrams which are not stopwords and which are stemmed

        self.set_of_unigrams = set()  # set of unigrams of the sentence
        self.set_of_bigrams = set()  # set of bigrams of the sentence

        self.set_of_processed_unigrams = set()  # set of processed unigrams of the sentence
        self.set_of_processed_bigrams = set()  # set of processed bigrams of the sentence

        self.length_of_raw_text = 0  # number of characters in the raw  of the sentence
        self.number_of_unigrams = 0  # number of tokens in the sentence
        self.number_of_bigrams = 0  # number of bigrams in the sentence
        self.number_of_processed_unigrams = 0  # number of processed tokens in the sentence
        self.number_of_processed_bigrams = 0  # number of processed bigrams in the sentence

        # dictionary which counts the number of occurrences of processed unigrams
        self.d_processed_unigram_count = defaultdict(int)
        # dictionary which counts the number of occurrences of processed bigrams
        self.d_processed_bigram_count = defaultdict(int)

    def set_text(self, text):
        """
        The method to use if you want that Sentence to be useful.
        :param text: unicode
        :return:
        """
        self.raw_text = text
        self.list_of_unigrams = word_tokenize(text)
        self.number_of_unigrams = len(self.list_of_unigrams)
        self.list_of_bigrams = get_bigrams(text)
        self.number_of_bigrams = len(self.list_of_bigrams)
        self.list_of_processed_unigrams = [stemmer.stem(uni.lower()) for uni in self.list_of_unigrams
                                           if uni.lower() not in stoplist and re.search(r'[a-zA-Z0-9]', uni) is not None]
        for uni in self.list_of_processed_unigrams:
            self.d_processed_unigram_count[uni] += 1

        #  should we use the bigrams of self.list_of_processed_unigrams or make bigrams differently
        # (as proposed above)
        for i in range(len(self.list_of_processed_unigrams) - 1):
            bigram = []
            for k in range(i, i + 2):
                bigram.append(self.list_of_processed_unigrams[k].lower())

            # do not consider bigrams containing punctuation marks
            marks = [t for t in bigram if not re.search('[a-zA-Z0-9]', t)]
            if len(marks) > 0:
                continue

            # do not consider ngrams composed of only stopwords
            stops = [t for t in bigram if t in stoplist]
            if len(stops) != 0:  # == len(bigram):
                continue

            # stem the ngram
            # bigram = [stemmer.stem(t) for t in bigram]

            self.list_of_processed_bigrams.append(tuple(bigram))
            self.d_processed_bigram_count[tuple(bigram)] += 1

        self.set_of_unigrams.update(self.list_of_unigrams)
        self.set_of_bigrams.update(self.list_of_bigrams)
        self.set_of_processed_unigrams.update(self.list_of_processed_unigrams)
        self.set_of_processed_bigrams.update(self.list_of_processed_bigrams)
        self.length_of_raw_text = len(self.raw_text)
        self.number_of_processed_unigrams = len(self.list_of_processed_unigrams)
        self.number_of_processed_bigrams = len(self.list_of_processed_bigrams)

    def __unicode__(self):
        return self.raw_text

    def __len__(self):
        return self.number_of_unigrams


class SentenceOfDocument(Sentence):
    """
    A sentence can be a part of a document.
    A relevant sentence is a sentence which has at least one bigram with document frequency larger than or equal to
    three.
    """
    def __init__(self):
        """

        :return:
        """
        Sentence.__init__(self)

        self.position_in_the_document = 0  # position of the sentence in the document it is extracted
        self.document_id = -1  # identifier of the document from where the sentence is

        # boolean value which states if the sentence is a relevant sentence
        self.is_relevant_sentence = False

        # list of BigramFeatures instances
        self.list_of_bigram_features = []
        # a SentenceFeature instance
        self.sentence_features = None

    def set_doc_id(self, doc_id):
        """
        identifier of the document from where the sentence is
        :param doc_id: integer
        :return:
        """
        self.document_id = doc_id

    def set_position_in_document(self, pos_doc):
        """
        position of the sentence in the document it is extracted
        :param pos_doc: integer
        :return:
        """
        self.position_in_the_document = pos_doc

    def set_relevant_sentence(self, document_frequency_for_bigrams):
        """

        :param document_frequency_for_bigrams: dictionary which comes from UpdateDocument.document_frequency_for_bigrams
        :return:
        """
        self.is_relevant_sentence = len([bigram for bigram in self.list_of_bigrams
                                         if document_frequency_for_bigrams[bigram] > 3]) != 0

    def compute_bigram_features(self, update_documents):
        """

        :param update_documents: an UpdateDocument instance
        :return:
        """
        for pos_in_sentence, bigram in enumerate(self.list_of_processed_bigrams):
            self.list_of_bigram_features.append(BigramFeatures(bigram, update_documents,
                                                               len(self.list_of_processed_bigrams), pos_in_sentence))

    def compute_sentence_features(self, update_documents, number_sentence_in_doc, summary_from_old_data):
        """

        :param update_documents: an UpdateDocument instance
        :param number_sentence_in_doc: integer
        :param summary_from_old_data:
        :return:
        """
        sentence_features = SentenceFeatures(update_documents.documents, update_documents.old_documents,
                                             update_documents.new_documents)
        sentence_features.update_feature_extraction_sentence(self, self.position_in_the_document,
                                                             number_sentence_in_doc, summary_from_old_data)
        self.sentence_features = sentence_features

    def get_vectorized_features(self):
        return self.sentence_features.get_vectorized_features()

    def cos_tfidf(self, other, complete):
        """

        :param other: instance of SummaryOfDocument
        :param doc:
        :return:
        """
        # only the sentence
        if isinstance(other, SummaryOfDocuments):
            bigrams = set()
            bigrams.update(self.set_of_processed_bigrams)
            bigrams.update(other.set_of_processed_bigrams)
            num = sum([self.d_processed_bigram_count[tuple(bigram)] *
                       other.d_processed_bigram_count[tuple(bigram)] *
                       (2./(delta(bigram in self.set_of_processed_bigrams)+delta(bigram in other.set_of_processed_bigrams)))**2
                       for bigram in bigrams])
            den_x = sum([(self.d_processed_bigram_count[tuple(bigram)] *
                         (1./(delta(bigram in self.set_of_processed_bigrams))))**2
                         for bigram in self.set_of_processed_bigrams])
            den_y = sum([(other.d_processed_bigram_count[tuple(bigram)] *
                        (1./(delta(bigram in other.set_of_processed_bigrams))))**2
                         for bigram in other.set_of_processed_bigrams])
            if den_x == 0 or den_y == 0:
                return 0
            return num/(math.sqrt(den_x)*math.sqrt(den_y))
        elif isinstance(other, SentenceOfDocument):
            bigrams = set()
            bigrams.update(self.set_of_processed_bigrams)
            bigrams.update(other.set_of_processed_bigrams)
            num = sum([self.d_processed_bigram_count[tuple(bigram)] *
                       other.d_processed_bigram_count[tuple(bigram)] *
                       math.log(float(len(complete.list_of_documents))/complete.document_frequency_for_bigrams[tuple(bigram)])**2
                       for bigram in bigrams])
            den_x = sum([self.d_processed_bigram_count[tuple(bigram)] *
                         math.log(float(len(complete.list_of_documents))/complete.document_frequency_for_bigrams[tuple(bigram)])**2
                         for bigram in self.set_of_processed_bigrams])
            den_y = sum([other.d_processed_bigram_count[tuple(bigram)] *
                         math.log(float(len(complete.list_of_documents))/complete.document_frequency_for_bigrams[tuple(bigram)])**2
                         for bigram in other.set_of_processed_bigrams])
            return num/(math.sqrt(den_x)*math.sqrt(den_y))
        else:
            print "strange data"
            print other


class Document:
    """
    The correct class of Document
    """

    def __init__(self):
        # list of Sentence instances
        self.list_of_sentences = []
        # raw form of the document
        self.raw_document = ""
        # list of raw sentences
        self.list_of_raw_sentences = []

        # set of processed and unprocessed of unigrams and bigrams
        self.set_of_unigrams = set()
        self.set_of_processed_unigrams = set()
        self.set_of_bigrams = set()
        self.set_of_processed_bigrams = set()

        # the position of the document in the set of documents it is
        self.doc_id = -1

        # some counts about the document
        self.number_of_sentences = 0
        self.number_of_unigrams = 0
        self.number_of_bigrams = 0
        self.number_of_processed_unigrams = 0
        self.number_of_processed_bigrams = 0

        # dictionary which counts the number of occurrences of processed unigrams
        self.d_processed_unigram_count = defaultdict(int)
        # dictionary which counts the number of occurrences of processed bigrams
        self.d_processed_bigram_count = defaultdict(int)

    def set_doc_id(self, doc_id):
        """
        set the identifier of the document from where the sentence is
        :param doc_id: integer
        :return:
        """
        self.doc_id = doc_id

    def set_text(self, text):
        """

        :param text: unicode or str
        :return:
        """
        self.raw_document = text
        self.list_of_raw_sentences = sent_detector.tokenize(text)

        # Process sentences
        for i, raw_sentence in enumerate(self.list_of_raw_sentences):
            sent = SentenceOfDocument()
            sent.set_position_in_document(i)
            sent.set_doc_id(self.doc_id)
            sent.set_text(raw_sentence)
            self.list_of_sentences.append(sent)
        self.number_of_sentences = len(self.list_of_sentences)

        # Update
        for sent in self.list_of_sentences:
            self.set_of_unigrams.update(sent.set_of_unigrams)
            self.set_of_processed_unigrams.update(sent.set_of_processed_unigrams)
            self.set_of_bigrams.update(sent.set_of_bigrams)
            self.set_of_processed_bigrams.update(sent.set_of_processed_bigrams)
            for key in sent.d_processed_unigram_count.keys():
                self.d_processed_unigram_count[key] += sent.d_processed_unigram_count[key]
            for key in sent.d_processed_bigram_count.keys():
                self.d_processed_bigram_count[key] += sent.d_processed_bigram_count[key]
            self.number_of_unigrams += sent.number_of_unigrams
            self.number_of_bigrams += sent.number_of_bigrams
            self.number_of_processed_unigrams += sent.number_of_processed_unigrams
            self.number_of_processed_bigrams += sent.number_of_processed_bigrams

    def set_relevant_sentences(self, document_frequency_for_bigrams):
        for sent in self.list_of_sentences:
            sent.set_relevant_sentence(document_frequency_for_bigrams)

    def __unicode__(self):
        return self.raw_document

    def main(self):
        pass

    def find_sentence(self, raw_text):
        for sent in self.list_of_sentences:
            if sent.raw_text == raw_text:
                return sent


class SetOfDocuments:
    def __init__(self):
        # list of Document instances
        self.list_of_documents = []

        # set of unigrams from all the documents
        self.set_of_unigrams = set()
        # set of bigrams from all the documents
        self.set_of_bigrams = set()
        # set of processed unigrams from all the documents
        self.set_of_processed_unigrams = set()
        # set of processed bigrams from all the documents
        self.set_of_processed_bigrams = set()

        # quite useless actually
        self.old_or_new = ""  # "new" or "old"

        # number of, processed of not, unigrams and bigrams of the set of documents
        self.number_of_unigrams = 0
        self.number_of_bigrams = 0
        self.number_of_processed_unigrams = 0
        self.number_of_processed_bigrams = 0

        # dictionary of number of occurrences of each unigram in the set of documents
        self.d_processed_unigram_count = defaultdict(int)
        # dictionary of number of occurrences of each unigram in the set of documents
        self.d_processed_bigram_count = defaultdict(int)

        # dictionary of number of documents each unigram in the set of documents appears
        self.document_frequency_for_bigrams = defaultdict(int)
        # dictionary of number of documents each bigram in the set of documents appears
        self.document_frequency_for_unigrams = defaultdict(int)

        self.document_frequency_for_processed_bigrams = defaultdict(int)
        self.document_frequency_for_processed_unigrams = defaultdict(int)

        self.reference_summary = None

    def set_old_or_new(self, category):
        """
        Useless I guess
        :param category:
        :return:
        """
        if category in ["old", "new"]:
            self.old_or_new = category
        else:
            print 'The category of the set of documents should be "old" or "new".'
            raise Warning

    def load_documents(self, list_of_text):
        """
        where the
        :param list_of_text:
        :return:
        """
        # Load
        for doc_id, text in enumerate(list_of_text):
            doc = Document()
            doc.set_text(text)
            doc.set_doc_id(doc_id)
            self.list_of_documents.append(doc)

        # Update
        for doc in self.list_of_documents:
            self.set_of_unigrams.update(doc.set_of_unigrams)
            self.set_of_processed_unigrams.update(doc.set_of_processed_unigrams)
            self.set_of_bigrams.update(doc.set_of_bigrams)
            self.set_of_processed_bigrams.update(doc.set_of_processed_bigrams)
            for key in doc.d_processed_unigram_count.keys():
                self.d_processed_unigram_count[key] += doc.d_processed_unigram_count[key]
            for key in doc.d_processed_bigram_count.keys():
                self.d_processed_bigram_count[key] += doc.d_processed_bigram_count[key]
            self.number_of_unigrams += doc.number_of_unigrams
            self.number_of_bigrams += doc.number_of_bigrams
            self.number_of_processed_unigrams += doc.number_of_processed_unigrams
            self.number_of_processed_bigrams += doc.number_of_processed_bigrams
            self.set_document_frequency_for_unigrams()
            self.set_document_frequency_for_bigrams()
            self.set_document_frequency_for_processed_unigrams()
            self.set_document_frequency_for_processed_bigrams()

    def load_summary(self, summary):
        """

        :param summary: list of SentenceOfDocuments instances
        :return:
        """
        summ = Summary()
        summ.load_summary(summary)
        self.reference_summary = summ

    def load_one_more_document(self, text):
        """
        Add one document to the set of documents, update all attributes
        :param text:
        :return:
        """
        # Adding one document
        doc = Document()
        doc.set_text(text)
        self.list_of_documents.append(doc)

        # Udpate
        self.set_of_unigrams.update(doc.set_of_unigrams)
        self.set_of_processed_unigrams.update(doc.set_of_processed_unigrams)
        self.set_of_bigrams.update(doc.set_of_bigrams)
        self.set_of_processed_bigrams.update(doc.set_of_processed_bigrams)
        for key in doc.d_processed_unigram_count.keys():
            self.d_processed_unigram_count[key] += doc.d_processed_unigram_count[key]
        for key in doc.d_processed_bigram_count.keys():
            self.d_processed_bigram_count[key] += doc.d_processed_bigram_count[key]
        self.number_of_unigrams += doc.number_of_unigrams
        self.number_of_bigrams += doc.number_of_bigrams
        self.number_of_processed_unigrams += doc.number_of_processed_unigrams
        self.number_of_processed_bigrams += doc.number_of_processed_bigrams
        self.set_document_frequency_for_unigrams()
        self.set_document_frequency_for_bigrams()
        self.set_document_frequency_for_processed_unigrams()
        self.set_document_frequency_for_processed_bigrams()

    def set_document_frequency_for_bigrams(self):
        """

        :return:
        """
        for doc in self.list_of_documents:
            for bigram in doc.set_of_bigrams:
                self.document_frequency_for_bigrams[bigram] += 1

    def get_document_frequency_for_bigrams(self):
        if len(self.document_frequency_for_bigrams) == 0:
            self.set_document_frequency_for_bigrams()
        return self.document_frequency_for_bigrams

    def set_document_frequency_for_processed_bigrams(self):
        for doc in self.list_of_documents:
            for bigram in doc.set_of_processed_bigrams:
                self.document_frequency_for_processed_bigrams[bigram] += 1

    def get_document_frequency_for_processed_bigrams(self):
        if len(self.document_frequency_for_processed_bigrams) == 0:
            self.set_document_frequency_for_processed_bigrams()
        return self.document_frequency_for_processed_bigrams

    def set_document_frequency_for_unigrams(self):
        for doc in self.list_of_documents:
            for unigram in doc.set_of_processed_unigrams:
                self.document_frequency_for_unigrams[unigram] += 1

    def get_document_frequency_for_unigrams(self):
        if len(self.document_frequency_for_unigrams) == 0:
            self.set_document_frequency_for_unigrams()
        return self.document_frequency_for_unigrams

    def set_document_frequency_for_processed_unigrams(self):
        for doc in self.list_of_documents:
            for unigram in doc.set_of_processed_unigrams:
                self.document_frequency_for_processed_unigrams[unigram] += 1

    def get_document_frequency_for_processed_unigrams(self):
        if len(self.document_frequency_for_processed_unigrams) == 0:
            self.set_document_frequency_for_processed_unigrams()
        return self.document_frequency_for_processed_unigrams

    def jaccard_similarity_processed_bigrams_text_and_summary(self):
        """

        :return:
        """
        return float(len(self.set_of_processed_bigrams.intersection(self.reference_summary.set_of_processed_bigrams)))/\
               len(self.set_of_processed_bigrams.union(self.reference_summary.set_of_processed_bigrams))

    def set_relevant_sentences(self):
        for doc in self.list_of_documents:
            doc.set_relevant_sentences(self.document_frequency_for_bigrams)

        return [sent for doc in self.list_of_documents for sent in doc.list_of_sentences
                if len([bigram for bigram in sent.list_of_processed_bigrams
                        if self.document_frequency_for_bigrams[bigram] >= 3]) >= 1]

    def shift_doc_ids(self, shift):
        """
        Useful for update summarization
        :param shift:
        :return:
        """
        for i, doc in enumerate(self.list_of_documents):
            doc.set_doc_id(shift + i)

    def __add__(self, other):
        """
        This is not commutative : the first argument must be the "old" SetOfDocuments and
        the second argument must be the "new" argument.
        :param other:
        :return:
        """
        new_doc = SetOfDocuments()
        # Combines all the lists and sets
        new_doc.list_of_documents.extend(self.list_of_documents)
        new_doc.list_of_documents.extend(other.list_of_documents)

        new_doc.set_of_unigrams.update(self.set_of_unigrams)
        new_doc.set_of_unigrams.update(other.set_of_unigrams)

        new_doc.set_of_processed_unigrams.update(self.set_of_processed_unigrams)
        new_doc.set_of_processed_unigrams.update(other.set_of_processed_unigrams)

        new_doc.set_of_bigrams.update(self.set_of_bigrams)
        new_doc.set_of_bigrams.update(other.set_of_bigrams)

        new_doc.set_of_processed_bigrams.update(self.set_of_processed_bigrams)
        new_doc.set_of_processed_bigrams.update(other.set_of_processed_bigrams)

        for key in self.d_processed_unigram_count.keys():
            new_doc.d_processed_unigram_count[key] += self.d_processed_unigram_count[key]
        for key in other.d_processed_unigram_count.keys():
            new_doc.d_processed_unigram_count[key] += other.d_processed_unigram_count[key]

        for key in self.d_processed_bigram_count.keys():
            new_doc.d_processed_bigram_count[key] += self.d_processed_bigram_count[key]
        for key in other.d_processed_bigram_count.keys():
            new_doc.d_processed_bigram_count[key] += other.d_processed_bigram_count[key]

        new_doc.number_of_unigrams += self.number_of_unigrams
        new_doc.number_of_unigrams += other.number_of_unigrams

        new_doc.number_of_bigrams += self.number_of_bigrams
        new_doc.number_of_bigrams += other.number_of_bigrams

        new_doc.number_of_processed_unigrams += self.number_of_processed_unigrams
        new_doc.number_of_processed_unigrams += other.number_of_processed_unigrams

        new_doc.number_of_processed_bigrams += self.number_of_processed_bigrams
        new_doc.number_of_processed_bigrams += other.number_of_processed_bigrams

        new_doc.set_document_frequency_for_unigrams()
        new_doc.set_document_frequency_for_bigrams()
        new_doc.set_document_frequency_for_processed_unigrams()
        new_doc.set_document_frequency_for_processed_bigrams()

        new_doc.reference_summary = other.reference_summary

        # Reassigns the doc_id of the right argument
        for i, doc in enumerate(self.list_of_documents):
            new_doc.list_of_documents[i].set_doc_id(i)
        return new_doc

    def __str__(self):
        text = ""
        for doc_text in self.list_of_documents:
            text += "\n\n" + str(doc_text)
        return text

    def find_sentence(self, raw_text):
        for doc in self.list_of_documents:
            for sent in doc.list_of_sentences:
                if sent.raw_text == raw_text:
                    return sent
        return None


class Summary:
    def __init__(self):
        # list of Sentence instances
        self.list_of_sentences_of_documents = []
        # list of strings
        self.list_of_raw_text = []
        # list of tokenized words
        self.list_of_tokenized_words = []

        self.set_of_unigrams = set()
        self.set_of_processed_unigrams = set()
        self.set_of_bigrams = set()
        self.set_of_processed_bigrams = set()

        self.d_processed_bigram_count = defaultdict(int)

    def load_summary(self, list_of_sentence):
        self.list_of_raw_text = list_of_sentence
        for raw_sent in list_of_sentence:
            sent = Sentence()
            sent.set_text(raw_sent)
            self.list_of_sentences_of_documents.append(sent)
        for sent in self.list_of_sentences_of_documents:
            for bigram in sent.list_of_processed_bigrams:
                self.d_processed_bigram_count[tuple(bigram)] += 1

    def __len__(self):
        return sum([len(sent) for sent in self.list_of_sentences_of_documents])

    def jaccard_similarity_processed_bigrams(self, other):
        assert isinstance(other, Summary)
        return float(len(self.set_of_processed_bigrams.intersection(other.set_of_processed_bigrams)))/\
               len(self.set_of_processed_bigrams.union(other.set_of_processed_bigrams))


class SummaryOfDocuments:
    """
    Kind of document, but has a different meaning, it's used as a list of SentenceOfDocument
    """
    def __init__(self):
        # list of SentenceOfDocument instances
        self.list_of_sentences_of_documents = []

        # list of strings
        self.list_of_raw_text = []

        # list of tokenized words
        self.list_of_unigrams = []

        self.set_of_unigrams = set()
        self.set_of_processed_unigrams = set()
        self.set_of_bigrams = set()
        self.set_of_processed_bigrams = set()

        self.d_processed_bigram_count = defaultdict(int)

    def load_summary(self, list_of_sent):
        """

        :param list_of_sent: list of instances of SentenceOfDocument
        :return:
        """
        self.list_of_sentences_of_documents = list_of_sent
        self.list_of_raw_text = [sent.raw_text for sent in list_of_sent]

        for sent in self.list_of_sentences_of_documents:
            self.list_of_unigrams.extend(sent.list_of_unigrams)
        for sent in self.list_of_sentences_of_documents:
            self.set_of_unigrams.update(sent.set_of_unigrams)
            self.set_of_processed_unigrams.update(sent.set_of_processed_unigrams)
            self.set_of_bigrams.update(sent.set_of_bigrams)
            self.set_of_processed_bigrams.update(sent.set_of_processed_bigrams)

        for sent in list_of_sent:
            for bigram in sent.list_of_processed_bigrams:
                self.d_processed_bigram_count[tuple(bigram)] += 1

    def __len__(self):
        """
        Number of sentences in the summary
        :return:
        """
        return len([sent.number_of_unigrams for sent in self.list_of_sentences_of_documents])


class UpdateDocument:
    """
    Class whose objective is to represent a document which has two parts : one old and one new.
    The old part is all the documents published before a time t and the new part is all the documents published after
    that time t. This is for update summarisation processing
    """
    def __init__(self, old_document, new_documents):
        # The part already summarized
        assert isinstance(old_document, SetOfDocuments)
        self.old_documents = old_document
        # The part that should be summarized taking into account what has already been summarized
        assert isinstance(new_documents, SetOfDocuments)
        self.new_documents = new_documents
        # The complete document
        self.documents = old_document + new_documents

        # identifier of the instance
        self.id = ""

        # set of relevant instances of Sentence
        self.relevant_sentences = []
        # number of bigrams in relevant_sentences
        self.number_of_bigrams_in_relevant_sentences = 0

        self.bigram_frequency_in_relevant_sentences = defaultdict(int)

    def set_relevant_sentences(self):
        # self.relevant_sentences =
        self.documents.set_relevant_sentences()

    def get_relevant_sentences(self):
        if len(self.relevant_sentences) == 0:
            self.set_relevant_sentences()
            for doc in self.documents.list_of_documents:
                for sent in doc.list_of_sentences:
                    if sent.is_relevant_sentence:
                        self.relevant_sentences.append(sent)
                        self.number_of_bigrams_in_relevant_sentences += len(sent.list_of_bigrams)
        return self.relevant_sentences

    def compute_bigram_frequency_in_relevant_sentences(self):
        for doc in self.documents.list_of_documents:
            for sent in doc.list_of_sentences:
                if sent.is_relevant_sentence:
                    for bigram in sent.list_of_bigrams:
                        self.bigram_frequency_in_relevant_sentences[
                            bigram] += 1. / self.number_of_bigrams_in_relevant_sentences

    def get_bigram_frequency_in_relevant_sentences(self):
        if len(self.bigram_frequency_in_relevant_sentences) == 0:
            self.compute_bigram_frequency_in_relevant_sentences()
        return self.bigram_frequency_in_relevant_sentences

    def compute_bigram_features(self):
        # if len(self.old_concept_based_summary) == 0:
        #     self.compute_old_concept_based_summary()
        for doc in self.documents.list_of_documents:
            for sent in doc.list_of_sentences:
                sent.compute_bigram_features(self)

    def compute_sentence_features(self):
        for doc in self.documents.list_of_documents:
            for sent in doc.list_of_sentences:
                sent.compute_sentence_features(self, doc.number_of_sentences)

    def compare_summaries(self):
        self.old_documents.reference_summary.jaccard_similarity_processed_bigrams(self.new_documents.reference_summary)


def delta(predicate):
    if predicate:
        return 1.
    else:
        return 0.


def jaccard_similarity(set1, set2):
    assert type(set1) == set
    assert type(set2) == set
    return float(len(set1.intersection(set2)))/len(set1.union(set2))


def get_bigrams(sentence):
    words = word_tokenize(sentence)
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    return bigrams


if __name__ == "__main__":
    pass

