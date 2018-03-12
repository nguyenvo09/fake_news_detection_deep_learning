import sys
import numpy as np
import json
import scrapy
from scrapy.crawler import CrawlerProcess
import requests
import time
import os
from selenium import webdriver
from shutil import copyfile
import datetime
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib
from os import listdir
from os.path import isfile, join
import pandas as pd
import random
import os
import requests
import nltk
import math
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# reload(sys)
# sys.setdefaultencoding('utf-8')
from nltk.tokenize import sent_tokenize

def inverse_document_frequencies(tokenized_documents):
    all_tokens_set = {}
    for url, doc in tokenized_documents.iteritems():
        for w in doc:
            assert w not in stop_words
            all_tokens_set[w] = 1
    print("here")
    idf_values = {}
    # all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    print(len(all_tokens_set) * len(tokenized_documents))

    dict_token_in_docs = {}
    for url, doc in tokenized_documents.iteritems():
        assert type(doc) == dict
        for w in doc:
            dict_token_in_docs[w] = dict_token_in_docs.get(w, 0) + 1

    for w in dict_token_in_docs:
        cnt_doc_contain_token = dict_token_in_docs[w]
        idf_values[w] = 1 + math.log(len(tokenized_documents) / (cnt_doc_contain_token))
    print("DONE")
    # for tkn in all_tokens_set:
    #     print("Token: ", tkn)
    #     cnt_doc_contain_token = 0
    #     for url, doc in tokenized_documents.iteritems():
    #         if tkn in doc:
    #             cnt_doc_contain_token += 1
    #     # contains_token = map(lambda doc: tkn in doc, tokenized_documents)
    #     idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (cnt_doc_contain_token))
    return idf_values


def compute_tfidf(dict_documents):
    print("Starting computing tfidf....: ", len(dict_documents))
    # tokenized_documents = [tokenize(d) for d in documents]
    idf_all_vocab = inverse_document_frequencies(dict_documents)
    print("ending inverse....")
    # print(idf_all_vocab.keys()[5])
    tfidf_for_words = {}
    list_tfidf_in_all_documents = {}
    for url, document in dict_documents.iteritems():
        for w in document:
            list_tfidf_in_all_documents[w] = list_tfidf_in_all_documents.get(w, [])
            tf = document[w]
            tfidf_value_of_this_term_in_this_doc = tf * idf_all_vocab[w]
            list_tfidf_in_all_documents[w].append(tfidf_value_of_this_term_in_this_doc)
        # doc_tfidf = {}
        # # print(document)
        # for term in idf_all_vocab.keys():
        #     tf = document.get(term, 0)
        #     # print(tf)
        #     # assert term not in list_tfidf_in_all_documents
        #     list_tfidf_in_all_documents[term] = list_tfidf_in_all_documents.get(term, [])
        #     tfidf_value_of_this_term_in_this_doc = tf * idf_all_vocab[term]
        #     list_tfidf_in_all_documents[term].append(tfidf_value_of_this_term_in_this_doc)
        #     # print(list_tfidf_in_all_documents[term])

    mean_tfidf_of_tokens = {}
    for token, list_tfidf in list_tfidf_in_all_documents.iteritems():
        # print(list_tfidf)
        # assert len(list_tfidf) == len(idf_all_vocab), '%s vs. %s' % (len(list_tfidf) ,len(idf_all_vocab) )
        mean_tfidf_of_tokens[token] = np.sum(list_tfidf) / float(len(idf_all_vocab))

    sorted_tfidf = sorted(mean_tfidf_of_tokens.items(), key=lambda x:x[1], reverse=True)
    print("Ending computing tfidf......")
    return sorted_tfidf

def process_urls_content(url_content_folder, parent):


    documents = [fn for fn in listdir(url_content_folder) if fn.endswith('.txt')]
    print("Numober of urls in this fold: ", len(documents))
    dict_documents = {}
    for fn in documents:
        p = join(url_content_folder, fn)
        fin = open(p, 'r')
        url_name = fn
        url_name = url_name.replace('\n', '')
        lines = []
        ignored = ["Sign up for the Snopes.com newsletter and get daily updates on all the best rumors, news and legends delivered straight to your inbox.",
                   "Know of a rumor you want investigated? Press related inquiry? Lonely and just want to chat?",
                   "Select from one of these options to get in touch with us:",
                   "We are experiencing some issues with our feedback form. To reach us in the interim, please email contact@teamsnopes.com.",
                   "Got a tip or a rumor? Contact us here.",
                   "We are experiencing some issues with our forms. Our development team is working on a solution.",
                   "Ask. Chat. Poke."]
        for line in fin:
            line = line.replace('\n', '')
            if line in ignored:
                print("yes")
                continue
            if line == "Fact Checker:":
                break
            ll = []
            for x in line:
                try:
                    x.encode('utf-8')
                    ll.append(x)
                except:
                    pass
            # print(ll)
            line = ''.join(ll)
            # line = line.encode('utf-8')

            line = line.lower().strip()
            lines.append(line + " ")
        all = ' '.join(lines)
        tokens = nltk.word_tokenize(all)
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= 2]
        # print(tokens)
        #remove stop words first to reduce complexity:
        doc_dict = {}
        for t in tokens:
            doc_dict[t] = doc_dict.get(t, 0) + 1
        assert '\n' not in tokens
        assert url_name not in dict_documents
        # assert 'http://' in url_name or 'https://' in url_name, url_name
        dict_documents[url_name] = doc_dict

    out_top_8K_words_file = '%s/out_top_16K_words_file.txt' % parent
    top8000 = compute_tfidf(dict_documents)
    fout = open(out_top_8K_words_file, 'w')
    cnt = 0
    for w,tfidf_vl in top8000:
        try:
            fout.write('%s %s %s\n' % (cnt, w, tfidf_vl))
            cnt += 1
            if cnt == 16000:
                break
        except:
            print("DKM: ", w)


def selectTop8000Words_based_tfidf():
    url_content_folder = 'webscrapedtext'
    parent = '.'
    process_urls_content(url_content_folder, parent)


def sentence_tokenize(infoloder='webscrapedtext', outfolder='sentences_tokenized'):
    '''build sententences based on top 8000 selected words.
    The first line of each document is the URL
    '''
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    dict_selected_tokens = {}
    fin1 = open('out_top_16K_words_file.txt', 'r')
    for line in fin1:
        _, w, _ = line.split()
        dict_selected_tokens[w] = 1
    assert len(dict_selected_tokens) == 16000
    documents = [fn for fn in listdir(infoloder) if fn.endswith('.txt')]
    print("Numober of urls in this fold: ", len(documents))
    dict_documents = {}
    infolder2 = 'crawled_websites'
    ignored = [
        "Sign up for the Snopes.com newsletter and get daily updates on all the best rumors, news and legends delivered straight to your inbox.",
        "Know of a rumor you want investigated? Press related inquiry? Lonely and just want to chat?",
        "Select from one of these options to get in touch with us:",
        "We are experiencing some issues with our feedback form. To reach us in the interim, please email contact@teamsnopes.com.",
        "Got a tip or a rumor? Contact us here.",
        "We are experiencing some issues with our forms. Our development team is working on a solution.",
        "Ask. Chat. Poke."]
    for fn in documents:
        p = join(infoloder, fn)
        fin = open(p, 'r')
        fin2 = open(join(infolder2, '%s.html' % fn.split('.')[0]), 'r')
        url_name = fin2.readline().replace('\n', '')
        assert 'http' in url_name
        # url_name = url_name.replace('\n', '')
        lines = []
        for line in fin:
            line = line.replace('\n', '')
            if line in ignored:
                print("yes")
                continue
            if line == 'Fact Checker:':
                break
            ll = []
            for x in line:
                try:
                    x.encode('utf-8')
                    ll.append(x)
                except:
                    pass
            # print(ll)
            line = ''.join(ll)
            # line = line.encode('utf-8')

            line = line.lower().strip()
            lines.append(line + " ")
        all = ' '.join(lines)
        sent_tokenize_list = sent_tokenize(all)
        fout = open('%s/%s' % (outfolder, fn), 'w')
        fout.write('%s\n' % url_name)
        for sent in sent_tokenize_list:
            if sent == "":
                continue
            args = sent.split()
            res = []
            for a in args:
                if a in dict_selected_tokens:
                    res.append(a)
            if len(res) == 0:
                continue
            sent = ' '.join(res)
            fout.write('%s\n' % sent)

def divide_dataset_into_5parts(infolder= 'sentences_tokenized', outfolder='train_test_data'):
    '''
    We ramdomly divide dataset into 80% training and 20% for testing.
    We guarantee the distribution of two classes are same
    We repeat the above evaluation scheme five times and report the average accuracy.
    :return:
    '''
    documents = [fn for fn in listdir(infolder) if fn.endswith('.txt')]
    dict_url_index = {}
    for idx, fn in enumerate(documents):
        p = join(infolder, fn)
        fin = open(p, 'r')
        url = fin.readline().replace('\n', '')
        # print(url)
        assert 'http' in url
        dict_url_index[url] = (p, fn)


    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    ground_truth = 'snopes_ground_truth.csv'
    df = pd.read_csv(ground_truth)
    df1 = df.loc[df['claim_label'] == True]
    df2 = df.loc[df['claim_label'] == False]
    s1 = zip(df1['snopes_page'], df1['claim_label'])
    s2 = zip(df2['snopes_page'], df2['claim_label'])
    # print(s1, len(s1))
    # print(s2, len(s2))
    assert len(s1) == len(s2)
    for i in xrange(5):
        p1 = '%s/data_%s' % (outfolder, i)
        if not os.path.exists(p1):
            os.mkdir(p1)

        train_folder = '%s/train' % p1
        test_folder = '%s/test' % p1
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)

        random.shuffle(s1)
        random.shuffle(s2)
        N = len(s1)
        first = int(0.8*N)
        train = s1[:first] + s2[:first]
        test = s1[first:N] + s2[first:N]
        print(len(train), len(test))

        for url, claim_label in train:
            assert url in dict_url_index, url
            src, name = dict_url_index[url]
            dest = '%s/%s' % (train_folder, name)
            copyfile(src, dest)

        for url, claim_label in test:
            assert url in dict_url_index
            src, name = dict_url_index[url]
            dest = '%s/%s' % (test_folder, name)
            copyfile(src, dest)

def process_wordcloud():
    fin = open('out_top_16K_words_file.txt', 'r')
    fout = open('top16K_words.csv', 'w')
    fout.write('idx,word,tfidf\n')
    for line in fin:
        idx, w, vl = line.split()
        fout.write('%s,%s,%s\n' % (idx, w, vl))


def genPathToRemove():
    base = 'hadoop fs -rm -R /user/nkvo/twitter_data/GeoData/2017/%s/%2d/@eaDir'
    for i in range(1, 13):
        for j in range(1, 32):
            print 'hadoop fs -rm -R /user/nkvo/twitter_data/GeoData/2017/%02d/%02d/@eaDir' % (i, j)


if __name__ == '__main__':
    print("TODO")
    # selectTop8000Words_based_tfidf()
    # sentence_tokenize()
    # divide_dataset_into_5parts()
    # process_wordcloud()
    genPathToRemove()