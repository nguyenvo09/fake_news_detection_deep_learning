import sys
import numpy as np
legitimate_websites=['snopes.com', 'factcheck.org', 'politifact.com', 'truthorfiction.com', 'opensecrets.org', 'slayer.com', 'slayer.net']
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
def trySelenimum(infile='', outfolder='crawled_friendship_DataB'):

    ''' Crawling follower following relationship on doesfollow.com website.
    Using selenium keke'''

    fin = open('snopes_ground_truth.csv', 'r')
    fin.readline()
    pages = []
    for line in fin:
        line = line.replace('\n', '')
        args = line.split(',')
        page = args[0]
        pages.append(page)
    assert len(pages) == 562
    # base_link = 'https://www.snopes.com/tag/fake-news'
    # links = ['%s/page/%s/' % (base_link, (i+1)) for i in xrange(50)]

    # driver.set_preference("browser.privatebrowsing.autostart", True)
    if not os.path.exists('crawled_websites'):
        os.mkdir('crawled_websites')
    cnt = 0
    for i in xrange(len(pages)):
        page = pages[i]
        try:
            print(page)
            response = requests.get(page)
            cnt += 1
            with open('crawled_websites/%s.html' % i, 'wb') as f:
                f.write('%s\n' % page)
                content = response.content
                f.write(content)
                # cnt += 1
            if cnt % 10 == 0:
                time.sleep(30)
        except:
            print "Unexpected error:", sys.exc_info()[0]




def start_cralwer(inFile='HOAXY_unique_URLs.txt'):
    fin = open(inFile, 'rb')
    URLs = []
    cnt = 0
    for line in fin:
        args = line.split()
        if len(args) == 3:
            continue
        cnt += 1
        if cnt <= 2778:
            continue
        URLs.append(args[0])

    cnt = 2778
    for url in URLs:
        cnt+=1
        try:
            print url
            response = requests.get(url)
            with open('crawled_websites/%s.html' % cnt, 'wb') as f:
                f.write('%s\n' % url)
                f.write(response.content)
            if cnt % 10 == 0:
                time.sleep(30)
        except:
            print "Unexpected error:", sys.exc_info()[0]
def get_links_each_page():
    fout = open('fake_news_link.txt', 'w')
    infiles = ['fake_news_page_%s.html' % i for i in xrange(1, 31)]
    dict_links = {}
    for infile in infiles:
        fin = open(infile, 'r')
        content = fin.read()
        soup = BeautifulSoup(content, 'html')
        # soup.find('article', )
        elems = soup.find_all("a", {"class": ["category-fake-news"]}, href=True)
        for e in elems:
            x = e['href']
            dict_links[x] = 1
            # print(e['href'])
            # print(e.get('href'))
    for url_ in dict_links:
        fout.write('%s\n' % (url_))


def stat_snope_dataset():
    # fin = open('snopes.csv', 'r')
    parts = pd.read_csv('snopes.csv', index_col=False)
    pages = zip(parts['snopes_page'], parts['claim_label'])
    dict_pages = {}
    true_news = {}
    fake_news = {}
    for p, ll in pages:
        if ll == 'true':
            true_news[p] = 1
        elif ll == 'false':
            fake_news[p] = 1

        if ll == 'false' or ll == 'true':
            dict_pages[p] = ll

    print(len(true_news), len(fake_news), len(dict_pages))

    # print(dict_pages.keys()[:5])
    # print(dict_pages.values()[:5])
    print(dict_pages.values().count('true'), dict_pages.values().count('false'))
    print(len(dict_pages))
    # x = fake_news[:281]
    # print(x[:5])
    fake_news = fake_news.keys()
    true_news = true_news.keys()
    random.shuffle(fake_news)
    fake_news = fake_news[:281]
    assert len(fake_news) == 281


    # print(fake_news[:5])
    assert len(true_news) == 281
    news = true_news + fake_news
    assert len(news) == 562, len(news)
    fin = open('snopes.csv', 'r')
    fout = open('snopes_ground_truth.csv', 'w')
    header = fin.readline()
    fout.write(header)
    dict_repeat = {}
    for line in fin:
        line = line.replace('\n', '')
        args = line.split(',')
        page = args[0]
        if page in news and page not in dict_repeat:
            dict_repeat[page] = 1
            fout.write('%s\n' % line)



if __name__ == '__main__':
    trySelenimum()
    # get_links_each_page()
    # stat_snope_dataset()