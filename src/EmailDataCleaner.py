import pandas as pd
import numpy as np
import pathlib
import os

from TextCleaner import TextCleaner
from UrlCleaner import get_domain
import EmailLogger
from EmailAnalyser import EmailAnalyser


class EmailDataCleaner:
    """Class pour nettoyer et standardiser les données"""

    def __init__(self, data=None):
        print('creating object')
        self.data = data

        self.theme = None  # dict

        self.corpus = None  # date/text (values = string of all words in date)
        self.dtm = None  # date/word (values = count)
        self.doc_urls_matrix = None  # date/urls (values = count)

    def remove_email_from_decrypteur(self, logpath=None):
        """return row without email from bouchera, alexis or jeff"""
        df = self.data
        index = None
        for word in ['jeff.yates', 'alexis.de.lancer', 'bouchra.ouatik']:
            word_index = df[df["from"].str.contains(word)]['from'].index
            if index is None:
                index = word_index
            else:
                index = np.concatenate([index, word_index])
        row_to_remove = df.loc[index]

        if logpath:
            df[df['from'].isin(row_to_remove['from'])].to_csv(logpath)
        else:
            df[df['from'].isin(row_to_remove['from'])].to_csv('removed_row_from_decrypteur.csv')
        self.data = df[~df['from'].isin(row_to_remove['from'])]

    def filter_by_start_date(self, yyyy=2020, m=3, d=1):
        """
        Filtre et conserve les données sup ou égale à la date.
        :param yyyy: année
        :param m: mois
        :param d: jour
        modifie self.data pour conserver seulement les courriels une date supérieur ou égal à yyyy-m-d
        """
        from datetime import datetime
        date = datetime(yyyy, m, d)
        filt = lambda x: x >= date
        self.data = self.data[self.data['datetime'].apply(filt)]

    def clean_urls(self):
        """ standardiser les urls
            - supprimer les urls provenants de domaine indésirable
            - regrouper par racine pour le top 20 des plus populaires
            - concerver seulement 5 urls par email au maximum
            """
        from UrlCleaner import UrlCleaner
        urlClean = UrlCleaner(self.data.urls)
        urlClean.remove_undesirable_domain()
        urlClean.standardise_youtube_url()
        urlClean.standardise_facebook_page_id()
        urlClean.remove_repetition_of_domain()

    def parse_dict_theme_from_csv(self, csv_path):
        """a partir d'un dataframe ou chaque colonne est un theme et chaque ligne/valeur est un mot du theme
        self.theme est simplement un dictionnaire où la key est le thème et la valeur est la liste de mots du thème
        """
        theme_raw = pd.read_csv(f"{csv_path}.csv")
        dict_theme = theme_raw.to_dict(orient='list')
        for theme in dict_theme.keys():
            dict_theme[theme] = [word for word in dict_theme[theme] if str(word) != 'nan']
        self.theme = dict_theme

    def add_theme(self):
        from TextCleaner import clean_text_for_analysis, word_in
        from tqdm import tqdm

        theme_array = []
        print('tagging theme to email')
        for idx in tqdm(self.data.index):
            email_theme = []
            text = self.data.loc[idx, 'text']
            text = clean_text_for_analysis(text)
            for theme, theme_words in zip(self.theme.keys(), self.theme.values()):
                in_list = any(word_in(w, text) for w in theme_words)
                if in_list:
                    email_theme.append(theme)
            theme_array.append(email_theme)
        self.data['theme'] = theme_array

    def add_domain(self):
        for idx in self.data.index:
            self.data.at[idx, 'domain'] = [get_domain(url) if url != '' else '' for url in self.data.loc[idx, 'urls']]

    def add_has_url(self):
        for idx in self.data.index:
            num_url = len(self.data.loc[idx, 'urls'])
            first_url = self.data.loc[idx, 'urls'][0]
            self.data.loc[idx, 'has_url'] = True if (num_url > 1 or first_url != '') else False

    def parse_text(self):
        text = []
        for idx in self.data.index:
            textCleaner = TextCleaner(body=self.data.loc[idx, 'body'], subject=self.data.loc[idx, 'subject'])
            textCleaner.create_text_from(attrs=['body', 'subject'])  # concat body and subject
            text.append(textCleaner.text)  # clean that text and append
        self.data['text'] = text

    def add_source(self):
        from EmailSourceFinder import EmailSourceFinder
        from tqdm import tqdm

        source_array = []
        most_common_domain = self.data.domain.explode().value_counts().drop('')
        print('tagging source to email')
        for idx in tqdm(self.data.index):
            sourceFinder = EmailSourceFinder(self.data.loc[idx])
            sourceFinder.add_source_from_email_domain()
            sourceFinder.add_source_from_common_domain_in_text(most_common_domain, top=50)
            sources = sourceFinder.source
            source_array.append(sources)

        self.data['source'] = source_array

    def add_theme_tag_from_url_scraper_data(self, scrap_url_df, domain):
        """regarde dans les urls scraper s'il y a des themes qui n'etait pas detecte seulement
        avec le texte du email"""
        log_data = []
        # scrap_url_df = scrap_url_df.set_index('url')
        for idx in self.data.index:
            theme_before = self.data.loc[idx, 'theme']
            # parcourir les emails
            theme_to_check = []
            # si un domain est facebook, voir les theme associes a l'url dans fb_url_df
            if domain in self.data.loc[idx, 'domain']:
                for url in self.data.loc[idx, 'urls']:
                    if url in scrap_url_df.index:
                        for theme in scrap_url_df.loc[url, 'theme']:
                            # aller chercher les themes trouves
                            theme_to_check.append(theme)
            # parmis les themes trouves, on ajoute seulement ceux qui ne sont pas deja la.
            for t in theme_to_check:
                if t not in self.data.loc[idx, 'theme']:
                    self.data.loc[idx, 'theme'].append(t)
            log_data.append([idx, theme_before, theme_to_check, self.data.loc[idx, 'theme']])
        return log_data

    def add_theme_manually(self, series):
        """pass a series of email_id and list of theme to correct theme"""
        if type(series.iloc[0]) != list:
            series = series.apply(lambda x: x.strip("[]").replace("'", "").strip().split(', '))
        self.data['theme'] = series

    def add_source_manually(self, series):
        """pass a series of email_id and list of theme to correct theme"""
        if type(series.iloc[0]) == str:
            series = series.apply(lambda x: x.strip("[]").replace("'", "").strip().split(', '))
        self.data['source'] = series

    def save(self, output_file):
        import pickle
        with open(f"{output_file}.pickle", 'wb') as f:
            pickle.dump(self, f)
        print(f'dataClean instance saved to pickle at {output_file}')

    @classmethod
    def from_pickle(cls, input_path):
        import pickle
        print(f'loading dataClean pickle from {input_path}')
        with open(f'{input_path}.pickle', 'rb') as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def from_csv(cls, filepath=None):
        print('loading from csv')
        if filepath is None:
            filepath = pathlib.PureWindowsPath(os.getcwd() + "\\output") / "email_raw.csv"

        df = pd.read_csv(filepath, index_col=0, parse_dates=['datetime'],
                         converters={
                             "attach_type": lambda x: x.strip("[]").replace("'", "").lower().strip().split(
                                 ", "),
                             "urls": lambda x: x.strip("[]").replace("'", "").strip().split(', ')
                         })
        # necessaire pour permettre le resampling dans EmailAnalyser
        df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))
        return EmailDataCleaner(data=df)


if __name__ == "__main__":
    theme_path = pathlib.PureWindowsPath(os.getcwd()) / "theme_words"
    email_raw_path = pathlib.PureWindowsPath(os.getcwd()) / "output" / 'csv_file' / 'email_raw.csv'
    dataCleaner = EmailDataCleaner.from_csv(filepath=email_raw_path)
    dataCleaner.remove_email_from_decrypteur()
    dataCleaner.clean_urls()
    dataCleaner.add_domain()
    dataCleaner.add_has_url()
    dataCleaner.parse_text()
    dataCleaner.parse_dict_theme_from_csv(theme_path)
    dataCleaner.add_theme()
    dataCleaner.add_source()

    print(dataCleaner.data.head())

    cur_dir = pathlib.PureWindowsPath(os.getcwd())
    dataCleaner.save(cur_dir / 'test_pickle')

    dataC2 = EmailDataCleaner.from_pickle(cur_dir / 'test_pickle')
    print(dataC2.data.head())

    emailAnalyser = EmailAnalyser(dataCleaner.data)
    df = emailAnalyser.get_email_with_theme(['virus'])
    df.to_clipboard()
    print(df)
    # emailAnalyser.grouby_word(['mot1', 'mot2']) #TODO ca serait cool de sortir tous les emails avec [X,X,X mots]
    # emailAnalyser.grouby_theme(['theme1, theme2, etc']) #TODO ca serait cool de sortir tous les emails avec [X,X,X theme]
