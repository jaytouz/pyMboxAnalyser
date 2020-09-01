"""YoutubeScrapper.py: Class pour scrapper les urls de Youtube.

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"

"""
import pytube
import requests
import re
from bs4 import BeautifulSoup
import TextCleaner


class YoutubeScrapper:
    """Scrapper pour la liste d'urls de Youtube passés dans le constructeur. Tag les urls en même temps selon le
    dictonnaire de theme"""
    def __init__(self, url, dict_theme):
        self.url = url
        self.dict_theme = dict_theme
        try:
            self.pytube_inst = pytube.YouTube(url)
        except Exception:
            self.pytube_inst = None
        self.title = None
        self.views = None
        self.author = None
        self.description = None
        self.rating = None
        self.date_published = None
        self.theme = None

    def parse_title(self):
        if self.pytube_inst:
            title = self.pytube_inst.title
            title = TextCleaner.standardize_text(title)
            self.title = title

    def parse_views(self):
        if self.pytube_inst:
            self.views = self.pytube_inst.views

    def parse_author(self):
        if self.pytube_inst:
            self.author = self.pytube_inst.author

    def parse_description(self):
        if self.pytube_inst:
            description = self.pytube_inst.description
            description = TextCleaner.standardize_text(description)
            self.description = description

    def parse_rating(self):
        if self.pytube_inst:
            self.rating = self.pytube_inst.rating

    def parse_date_published(self):
        if self.pytube_inst:
            req = None
            soup = None
            date = None
            try:
                req = requests.get(self.url)
            except Exception as e:
                print(e, "FOR : ", self.url)
            if req:
                soup = str(BeautifulSoup(req.content, features="lxml"))
            if soup:
                pattern = '"datePublished"/>\n<meta content="\d{4}-\d{2}-\d{2}"'
                res = re.findall(pattern, str(soup))
                if res:
                    date = res[0].split('="')[-1].strip('"')
            self.date_published = date

    def parse_data(self, attrs=None):
        if attrs is None:
            attrs = ['parse_title', 'parse_views', 'parse_author', 'parse_description',
                     'parse_date_published', 'parse_rating', 'add_theme']
        if attrs:
            for att in attrs:
                parser = getattr(self, att)
                parser()

    def add_theme(self):
        url_theme = []
        title = self.title if self.title else ''
        description = self.description if self.description else ''

        text = ' '.join([title, description])
        text = TextCleaner.clean_text_for_analysis(text)
        for theme, theme_words in zip(self.dict_theme.keys(), self.dict_theme.values()):
            in_list = any(TextCleaner.string_in_text(s, text) for s in theme_words)
            if in_list:
                url_theme.append(theme)
        self.theme = url_theme

    def export_to_list(self):
        return [self.url,
                self.title,
                self.date_published,
                self.description,
                self.author,
                self.rating,
                self.views,
                self.theme]

    def export_as_series(self):
        from pandas import Series
        index = ['url', 'title', 'date_published', 'description', 'author', 'rating', 'views', 'theme']
        data = self.export_to_list()
        return Series(data, index=index)
