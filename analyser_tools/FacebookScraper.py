import requests
import re
from bs4 import BeautifulSoup
import TextCleaner


class FacebookScraper:

    def __init__(self, url, dict_theme):
        self.soup = None
        self.dict_theme = dict_theme
        self.parse_html(url)
        self.url = url
        self.commment_count = None
        self.reaction_count = None
        self.share_count = None
        self.title = None
        self.date = None
        self.view_count = None
        self.theme = None

    def parse_html(self, url):
        req = None
        try:
            req = requests.get(url)
        except Exception as e:
            print(e, "FOR : ", url)
        if req:
            self.soup = str(BeautifulSoup(req.content, features="lxml"))
        else:
            self.soup = ''

    def parse_comment_count(self):
        comment_parttern = ['comment_count:{total_count:\d+}', "commentcount:\d+"]
        comment_count = None
        for pattern in comment_parttern:
            res = re.findall(pattern, self.soup)
            if res:
                comment_count = res[0].strip('{}').split(':')[-1]
        self.comment_count = int(comment_count) if comment_count is not None else None

    def parse_reaction_count(self):
        reaction_parttern = ['reaction_count:{count:\d+}', 'likecount:\d+']
        reaction_count = None
        for pattern in reaction_parttern:
            res = re.findall(pattern, self.soup)
            if res:
                reaction_count = res[0].strip('{}').split(':')[-1]
        self.reaction_count = int(reaction_count) if reaction_count is not None else None

    def parse_share_count(self):
        share_pattern = ['share_count:{count:\d+}']
        share_count = None
        for pattern in ['share_count:{count:\d+}']:
            res = re.findall(pattern, self.soup)
            if res:
                share_count = res[0].strip('{}').split(':')[-1]
        self.share_count = int(share_count) if share_count is not None else None

    def parse_title(self):
        title_pattern = '<title id="pageTitle">.+</title>'
        title = None
        res = re.findall(title_pattern, self.soup)
        if res:
            title = res[0].split('</')[0].split('>')[-1]
        self.title = title

    def parse_date(self):
        date_pattern = 'datePublished":"\d+-\d+[\-\+]{1}[A-Z0-9]+:\d+:\d+[\-\+]{1}\d+:\d+'
        date = None
        res = re.findall(date_pattern, self.soup)
        if res:
            date = res[0].split('":"')[-1]
        self.date = date if date is not None else None

    def parse_view(self):
        view_pattern = ['viewCount:"[\d,]+', 'postViewCount:"[\d,]+"']
        view_count = None
        for pattern in view_pattern:
            res = re.findall(pattern, self.soup)
            if res:
                view_count = res[0].strip('{}').split(':')[-1]
                if ',' in view_count:
                    view_count = re.sub('[,"]', '', view_count)
        self.view_count = int(view_count) if view_count is not None else None

    def parse_data(self, attrs=None):
        if attrs is None:
            attrs = ['parse_comment_count', 'parse_reaction_count', 'parse_share_count', 'parse_title',
                     'parse_date', 'parse_view', 'add_theme']
        if attrs:
            for att in attrs:
                parser = getattr(self, att)
                parser()

    def add_theme(self):
        url_theme = []
        if self.title:
            text = ' '.join([self.title])
            text = TextCleaner.clean_text_for_analysis(text)
            for theme, theme_words in zip(self.dict_theme.keys(), self.dict_theme.values()):
                in_list = any(TextCleaner.string_in_text(s, text) for s in theme_words)
                if in_list:
                    url_theme.append(theme)
        self.theme = url_theme

    def export_as_list(self):
        return [self.url,
                self.title,
                self.date,
                self.comment_count,
                self.reaction_count,
                self.share_count,
                self.view_count,
                self.theme]

    def export_as_series(self):
        from pandas import Series
        index = ['url', 'title', 'date', 'comment_count', 'reaction_count', 'share_count', 'view_count', 'theme']
        data = self.export_as_list()
        return Series(data, index=index)
