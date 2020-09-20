"""EmailDataFrame.py: Class principal. Chaque class est construite pour interagir avec un dataframe particulier, soit de
email, d'url, un corpus ou un document term matrix (les deux derniers sont pour l'analyse des mots dans le texte).

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"

"""



import logging
import pandas as pd
import numpy as np
from copy import deepcopy
import pathlib
import os

import UrlCleaner
import TextCleaner



class EmailDfBaseClass(object):
    """Cette class permet d'hériter des fonctions nécessaires de Pandas et qui seront communes à toutes les
    class de la forme DataFrame dans ce projet.

    Toutes les sous class de EmailDFBaseClass ont en commun d'avoir un attribut self.df qui est un
    pd.DataFrame

    Les différences des enfants sont le type de données en index et les colonnes (ex Url vs Email) ou
    les colonnes qu'on peut s'attendre à retrouver ex : (DTM vs Corpus)"""

    theme = None

    def __init__(self, df):
        self.df = df

    def head(self, val=5):
        """same as Pandas.DataFrame.head()"""
        return self.df.head(val)

    def tail(self, val=5):
        """same as Pandas.DataFrame.tail()"""
        return self.df.tail(val)

    @property
    def columns(self):
        """same as Pandas.DataFrame.columns"""
        return self.df.columns

    @property
    def index(self):
        """same as Pandas.DataFrame.index"""
        return self.df.index

    def filt(self, column: str, with_values: list, condition='and'):
        """retourne la meme class que l'instance d'ou l'appel provient avec seulement les courriels contenants
        les valeurs recherchées dans une colonne donnée. (possibilité de restreindre à un des thèmes ou tous les thèmes)

        :example
        EmailDfBaseClass.filt('theme', with_values=['complot', 'bilan'], condition = 'or')
            retourne un objectDF contenant toutes les lignes où ont retrouve complot ou bilan dans la colonne theme.
        """
        df = self.df.apply(deepcopy)  # needed because df contains python object

        condition_and = lambda values: all(v in values for v in with_values)  # si tous les themes demander
        condition_or = lambda values: any(v in values for v in with_values)  # si au moins un des thèmes demander
        condition = condition_and if condition == 'and' else condition_or  # choisir une des deux conditions
        df_output = df[df[column].apply(condition)]

        return type(self)(df_output)  # permet de retrourner une instance du meme type


    def to_clipboard(self):
        """same as Pandas.DataFrame.to_clipboard()"""
        self.df.to_clipboard()

    def to_csv(self, output_path=None, index = False):
        """output_path must be constructed with pathlib module, save the dataframe in the instance to csv"""
        self.df.head()
        self.df.to_csv(output_path, index=index)


    def to_pickle(self, output_file):
        import pickle
        with open(f"{output_file}.pickle", 'wb') as f:
            pickle.dump(self, f)
        print(f'{type(self)} instance saved to pickle at {output_file}')

    @classmethod
    def read_dict_theme_from_csv(self, csv_path):
        """a partir d'un dataframe ou chaque colonne est un theme et chaque ligne/valeur est un mot du theme
        self.theme est simplement un dictionnaire où la key est le thème et la valeur est la liste de mots du thème
        """
        try:
            theme_raw = pd.read_csv(f"{csv_path}")
        except UnicodeDecodeError:
            theme_raw = pd.read_csv(f"{csv_path}", encoding='iso-8859-1')

        dict_theme = theme_raw.to_dict(orient='list')
        for theme in dict_theme.keys():
            dict_theme[theme] = [word for word in dict_theme[theme] if str(word) != 'nan']
        EmailDfBaseClass.theme = dict_theme

    @classmethod
    def from_pickle(self, pickle_path):
        import pickle
        print(f'loading pickle from {pickle_path}')
        with open(f'{pickle_path}.pickle', 'rb') as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def from_csv(cls, csv_path, index_col = None, converters={}, parse_dates=['datetime']):
        """read dataframe from csv, converter are used to convert list-like cell from string format
        to list-like python object, parse_dates should be ['datetime'] unless if there's no column datetime"""

        df = pd.read_csv(csv_path, index_col=index_col, parse_dates=parse_dates, converters=converters)
        # necessaire pour permettre le resampling
        if parse_dates is not None and 'datetime' in parse_dates:
            df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))
        return cls(df)


class EmailDF(EmailDfBaseClass):
    """Class qui permet de manipuler le pd.DataFrame contenant un courriel par ligne
        Les fonctions de d'instance retourne generalement un objet de type EmailDataFrame afin
        d'enchaîner les commandes suivies de point :

        ex df_email.filt(column=theme, with_values['bilan', 'complot']).get_top_url()
        ceci permettrait donc de filtrer et ensuite d'appliquer une fonction d'extraction.

        """

    def __init__(self, df_email):
        super().__init__(df_email)

    @property
    def email_per_day(self):
        df = self.df.apply(deepcopy)  # needed because df contains python object
        return df.set_index('datetime').resample('D').count()['email_id']

    def filt_by_date(self, start_date: tuple, end_date: tuple):
        """
        permet de filtrer le df de l'instance entre les deux dates
        :param start_date: tuple with (yyyy,m,d)
        :param end_date: tuple with (yyyy,m,d)
        :return: instance instance of EmailDF with date between start_date and end_date inclusively
        """
        from datetime import datetime
        df = self.df.apply(deepcopy)  # needed because df contains python object

        start_yyyy, start_mm, start_day = start_date
        end_yyyy, end_mm, end_day = end_date

        datetime_start = datetime(start_yyyy, start_mm, start_day)
        filter_start = lambda x: x >= datetime_start
        datetime_end = datetime(end_yyyy, end_mm, end_day)
        filter_end = lambda x: x <= datetime_end

        df = df[df['datetime'].apply(filter_start)]
        df = df[df['datetime'].apply(filter_end)]

        return EmailDF(df_email=df)

    def remove_email_by_from(self, from_key_words=['jeff.yates', 'alexis.de.lancer', 'bouchra.ouatik'], logpath=None):
        """retire les email qui dans la colonne from contiennent une adresse courriel contenant exactement un des
        mots dans from_key_words"""
        df = self.df.apply(deepcopy)  # needed because df contains python object

        index = None
        for word in from_key_words:
            word_index = df[df["from"].str.contains(word)]['from'].index
            if index is None:
                index = word_index
            else:
                index = np.concatenate([index, word_index])
        row_to_remove = df.loc[index]

        if logpath:
            df[df['from'].isin(row_to_remove['from'])].to_csv(logpath)

        return EmailDF(df_email=df[~df['from'].isin(row_to_remove['from'])])

    def clean_urls(self):
        """ standardiser les urls
            - supprimer les urls provenants de domaine indésirable
            - regrouper par racine pour le top 20 des plus populaires
            - concerver seulement 5 urls par email au maximum
            """
        from UrlCleaner import UrlCleaner
        df = self.df.apply(deepcopy)  # needed because df contains python object
        urlClean = UrlCleaner(df.urls)
        urlClean.remove_undesirable_domain()
        urlClean.standardise_youtube_url()
        urlClean.standardise_facebook_page_id()
        urlClean.remove_repetition_of_domain()
        return EmailDF(df_email=df)

    def add_domain_column(self):
        """permet d'ajouter une colonne contenant le domain de l'url contenu dans la colonne urls."""
        from UrlCleaner import get_domain
        from tqdm import tqdm
        print("creating domain column")
        df = self.df.apply(deepcopy)  # needed because df contains python object
        for idx in tqdm(df.index):
            df.at[idx, 'domain'] = [get_domain(url) if url != '' else '' for url in df.loc[idx, 'urls']]
        return EmailDF(df_email=df)

    def add_has_url_column(self):
        """ajoute la colonne has_url. Cette valeur est True s'il a un url ou plus, False sinon"""
        from tqdm import tqdm

        print("creating has_url column")
        df = self.df.apply(deepcopy)  # needed because df contains python object
        for idx in tqdm(df.index):
            num_url = len(df.loc[idx, 'urls'])
            first_url = df.loc[idx, 'urls'][0]
            df.loc[idx, 'has_url'] = True if (num_url > 1 or first_url != '') else False
        return EmailDF(df_email=df)

    def add_text_column(self):
        """Combine la colonne body et subject pour en faire une colonne text.
         Le texte est standardisé (correction minimal)"""
        from TextCleaner import standardize_text
        from tqdm import tqdm
        print("creating text column")

        df = self.df.apply(deepcopy)  # needed because df contains python object

        text_cells_values = []
        for idx in tqdm(df.index):
            text = ' '.join([str(df.loc[idx, 'body']), str(df.loc[idx, 'subject'])])
            text = standardize_text(text)  # correct encoding, remove emoji, non-roman, extra space or tab or linebreak
            text_cells_values.append(text)  # clean that text and append
        df['text'] = text_cells_values
        return EmailDF(df_email=df)

    def add_theme_column(self):
        """Tag les courriels en fonction de la variable EmailDfBaseClass.theme. Recherche le mot, son féminin ou pluriel
        dans la colonne text. Avant la recherche du mot, le text est corrigé pour l'analyser, voir le module
         TextCleaner pour plus de détail sur la fonctionRetourne l'instance avec la nouvelle colonne de theme."""
        from TextCleaner import clean_text_for_analysis, string_in_text
        from tqdm import tqdm

        print('tagging theme to email')
        df = self.df.apply(deepcopy)  # needed because df contains python object
        theme_array = []
        for idx in tqdm(df.index):
            email_theme = []
            text = df.loc[idx, 'text']
            # retire certain mot commun dans les courriels, repetition de 5 chiffres +, lettre seul
            text = clean_text_for_analysis(text)
            for theme, theme_words in zip(self.theme.keys(),
                                          self.theme.values()):  # TODO tester pour voir si c'est accessible dans l'instance ou si je dois appeler la class

                in_list = any(string_in_text(s, text) for s in theme_words)
                if in_list:
                    email_theme.append(theme)
            theme_array.append(email_theme)
        df['theme'] = theme_array
        return EmailDF(df_email=df)

    def add_source_column(self):
        """Ajoute la colonne source qui est l'equivalent du domain, mais en gardant les valeurs uniques seulement"""

        print('tagging source to email')
        df = self.df.apply(deepcopy)  # needed because df contains python object
        df['source'] = df['domain']
        df['source'].apply(set).apply(list)
        return EmailDF(df_email=df)

    def groupby_theme(self, prop=True):

        data = {theme: self.filt('theme', with_values=[theme]).df.shape[0] for theme in
                self.theme.keys()}
        if prop:
            return pd.Series(data) /self.df.shape[0] * 100
        else:
            return pd.Series(data)

    def theme_per_day(self, pourc=True):
        """assemble les themes par jour et retour un dataframe avec comme index les journées,
        comme colonnes les themes et comme valeur la fréquence d'apparition de chaque thème dans la journeé"""
        df_per_day = self.df.set_index(pd.DatetimeIndex(self.df['datetime'])).resample('D').agg({'theme': sum})
        data = {theme: [] for theme in self.theme.keys()}
        for d in df_per_day.index:
            for theme in self.theme.keys():
                if df_per_day.loc[d, 'theme'] == 0:
                    count = 0
                else:
                    count = df_per_day.loc[d, 'theme'].count(theme)
                data[theme].append(count)

        if pourc:
            return pd.DataFrame(data, index=df_per_day.index).apply(lambda s : s/self.email_per_day * 100)

        else:
            return pd.DataFrame(data, index=df_per_day.index)

    def groupby_word_in_theme(self):
        """:return multiindex level 0 = theme, level 1 = word in theme, value = num_email with word in theme"""
        from TextCleaner import get_word_variation

        def multiindex_from_theme(theme):
            index = []
            for t in theme.keys():
                for word in theme[t]:
                    index.append((t, word))
            return pd.MultiIndex.from_tuples(index)

        multiindex = multiindex_from_theme(self.theme)
        data = {theme_word: self.filt('text', with_values=get_word_variation(
            theme_word[1]), condition='or').df.shape[0] for theme_word in multiindex}

        return pd.Series(data)

    def add_theme_manually(self, series):
        # TODO pas encore tester, c'est en prévision de l'ajout des courriels non tagger par le programme
        """pass a series of email_id and list of theme to correct theme"""
        if type(series.iloc[0]) != list:
            series = series.apply(lambda x: x.strip("[]").replace("'", "").strip().split(', '))
        self.data['theme'] = series

    def add_source_manually(self, series):
        # TODO pas encore tester, c'est en prévision de l'ajout des courriels non tagger par le programme
        """pass a series of email_id and list of theme to correct theme"""
        if type(series.iloc[0]) == str:
            series = series.apply(lambda x: x.strip("[]").replace("'", "").strip().split(', '))
        self.data['source'] = series

    def update_theme_using_urls_tag(self, url_df, url_from_domain):
        """regarde dans les urls scraper s'il y a des themes qui n'etait pas detecte seulement
        avec le texte du email

        :param url_df: UrlDF instance, obtenu d'un scrapper d'info d'urls
        :param url_from_domain: domain des urls scrapper avec des themes a regarder"""
        df = self.df.apply(deepcopy)  # needed because df contains python object
        for idx in df.index:
            theme_before = df.loc[idx, 'theme']
            # parcourir les emails
            theme_url = []
            # si un domain est facebook ou youtube, voir les theme associes a l'url dans url_df
            if url_from_domain in df.loc[idx, 'domain']:
                for url in df.loc[idx, 'urls']:
                    if url in url_df.index:
                        url_theme = url_df.df[url_df.df.url == url]['theme']
                        for theme in url_theme:
                            # aller chercher les themes trouves
                            theme_url.append(theme)
            # parmis les themes trouves, on ajoute seulement ceux qui ne sont pas deja la.
                df.at[idx,'theme'] = list(set(theme_before + theme_url))
                # print(theme_before==list(set(theme_before + theme_url)))
        return EmailDF(df_email=df)

    @classmethod
    def from_raw_csv(cls, csv_path=None):
        """from csv collected from mboxParser"""
        import pathlib
        import os

        if csv_path is None:
            filepath = pathlib.Path(os.getcwd() + "\\output") / "email_raw.csv"

        df = pd.read_csv(filepath, index_col=0, parse_dates=['datetime'],
                         converters={
                             "attach_type": lambda x: x.strip("[]").replace("'", "").lower().strip().split(
                                 ", "),
                             "urls": lambda x: x.strip("[]").replace("'", "").strip().split(', '),
                         })
        # necessaire pour permettre le resampling
        df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))
        return EmailDF(df_email=df)



class UrlDF(EmailDfBaseClass):
    """Class qui permet de manipuler le pd.DataFrame contenant un url par ligne
        Les fonctions de d'instance retourne generalement un objet de type UrlDF afin
        d'enchaîner les commandes suivies de point :

        ex df_url.filt(column=theme, with_values['bilan', 'complot']).get_most_viewed()
        ceci permettrait donc de filtrer et ensuite d'appliquer une fonction d'extraction.

        """

    def __init__(self, df_url):
        super().__init__(df_url)

    def update_theme(self, using_col = 'description'):
        """Tag les courriels en fonction de la variable EmailDfBaseClass.theme. Recherche le mot, son féminin ou pluriel
        dans la colonne using_col. Avant la recherche du mot, le text est corrigé pour l'analyser, voir le module
         TextCleaner pour plus de détail sur la fonctionRetourne l'instance avec la nouvelle colonne de theme.

         :param using_col: nom de la colonne a utiliser pour tagger les themes.
         """
        from TextCleaner import clean_text_for_analysis, string_in_text
        from tqdm import tqdm

        print('updating theme to urlDf')
        df = self.df.apply(deepcopy)  # needed because df contains python object
        theme_array = []
        for idx in tqdm(df.index):
            url_theme = []
            text = df.loc[idx, using_col]
            # retire certain mot commun dans les courriels, repetition de 5 chiffres +, lettre seul
            text = clean_text_for_analysis(str(text))
            for theme, theme_words in zip(self.theme.keys(),
                                          self.theme.values()):
                in_list = any(string_in_text(s, text) for s in theme_words)
                if in_list:
                    url_theme.append(theme)
            theme_array.append(url_theme)
        df['theme'] = theme_array
        return UrlDF(df_url=df)


class EmailCorpus(EmailDfBaseClass):
    """
    Cette class permet de manipuler un pd.DataFrame avec
    des emails comme rangées et une colonne contenant tout le texte.
    """

    def __init__(self, corpus_df, by, sampling):
        """
        initiation d'une instance de EmailCorpus
        :param corpus_df: pd.DataFrame une colonne text et des rangées d'email ou de temps selon l'unité
        """
        super().__init__(corpus_df)
        self.by = by
        self.sampling = sampling


    @classmethod
    def from_emailDF(cls, emailDF, by='email_id', sampling = 'D'):
        """
        Creation du corpus a partir d'un object EmailDF.
        :param emailDF: instance de EmailDF
        :param by: default = 'email_id', autre option, 'datetime'
        :param sampling: si by = datetime, sampling peut être 'D', 'M', 'Y', 'W' etc voir doc pd.resampling
        :return: EmailCorpus object
        """
        print('creating corpus from emailDf')
        df = emailDF.df.apply(deepcopy)  # needed because df contains python object
        if by == 'email_id':
            corpus = df[['email_id', 'text']].set_index('email_id')
            return EmailCorpus(corpus, by='email_id', sampling = sampling)

        elif by == 'datetime':
            corpus = df.set_index('datetime').resample(sampling).agg({'text': ' '.join})
            return EmailCorpus(corpus, by='email_id', sampling = sampling)




class EmailDTM(EmailDfBaseClass):
    """Cette class permet de manipuler un objet de type pd.DataFrame
    avec la forme d'un document term matrix. Donc chaque ligne correspond à un courriel, ou une unité de la
    fréquence d'échantillonnage choisis et chaque colonne correspond à un mot unique du corpus
    """

    def __init__(self, dtm_df, corpus_df):
        super().__init__(dtm_df)
        self.corpus_df = corpus_df

    @property
    def unique(self):
        """toutes les valeurs de 1 ou plus sont converti en 1, car on s'intéresse au jour ou au courriel ou le mot
        est apparu au moins une fois"""
        df = self.df.copy()
        df[df >= 1] = 1
        return EmailDTM(df, self.corpus_df)


    def proportion(self, email_per_day):
        return self.df.apply(lambda x: (x / email_per_day) * 100)


    def update_dtm_with_dict_theme(self):
        """ajoute les éléments du dict de theme qui sont composés de plusieurs mots."""
        #TODO vraiment pas efficient... faudrait creer un petit dataframe avec les nouvelles colonnes a ajouter et ensuite concat puis retourner l'instance de emailDTM

        for theme in self.theme.keys():
            for word in self.theme[theme]:
                if len(word.split()) > 1:
                    ngram = tuple(word.lower().split())
                    self.add_ngram_colunm(ngram)
        return EmailDTM(self.df, self.corpus_df)

    def add_ngram_colunm(self, ngram_tuple):
        """ajoute une colonne dans le dtm correspondant à une suite de mot"""
        from Errors import NGramTooLarge
        n = len(ngram_tuple) #
        if n > 5:
            raise NGramTooLarge
        col_name = ' '.join(ngram_tuple)
        data = []
        print("adding " + col_name + " to dtm")
        for idx in self.corpus_df.index:
            list_ngram = EmailDTM.list_word_ngram(n, self.corpus_df.loc[idx,'text'])
            count = list_ngram.count(ngram_tuple)
            data.append(count)
        self.df[col_name] = data


    @staticmethod
    def list_word_ngram(n, text):
        """
        Decoupe le texte en sequence de ngram. Retourne une liste de tuple
        :param n:
        :param text:
        :return:
        """
        import re
        words = re.findall('\w+', text.lower())
        if n == 2:
            list_ngram = zip(words, words[1:])
        elif n== 3:
            list_ngram = zip(words, words[1:], words[2:])
        elif n== 4:
            list_ngram = zip(words, words[1:], words[2:], words[3:])
        elif n == 5:
            list_ngram = zip(words, words[1:], words[2:], words[3:], words[4:])
        return list(list_ngram)

    @classmethod
    def from_corpus(cls, emailCorpus):
        """
        Creation emailDTM d'une instance de EmailCorpus
        :param emailCorpus:
        :return: EmailDTM
        """
        print("creating dtm from corpus")
        from stop_words import get_stop_words
        from sklearn.feature_extraction import text
        from sklearn.feature_extraction.text import CountVectorizer

        from TextCleaner import clean_text_for_analysis

        #Mots qui n'ajoute pas de sens dans une phrase en Francais ou en anglais
        french_stop_words = get_stop_words('french')
        english_stop_words = list(text.ENGLISH_STOP_WORDS)
        my_stop_words = french_stop_words + english_stop_words

        #s'assurer qu'aucun mot des themes n'est dans les stop words ex: état
        for t in emailCorpus.theme.keys():
            for w in emailCorpus.theme[t]:
                if w.lower() in my_stop_words:
                    my_stop_words.remove(w.lower())

        corpus = emailCorpus.df.copy() #deepcopy du DataFrame
        #nettoyer le text pour l'analyse
        corpus['text'] = corpus['text'].apply(lambda text :  clean_text_for_analysis(text))

        #creation de la matrix document terme (document term matrix)
        cv = CountVectorizer(stop_words=my_stop_words)
        data_cv = cv.fit_transform(corpus.text)
        dtm_df = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        dtm_df.index = corpus.index

        return EmailDTM(dtm_df, corpus)

if __name__ == '__main__':
    print('start')
    # VOIR NOTEBOOK POUR UTILISATION


    # cwd = pathlib.PurePath(os.getcwd())
    # project_root_dir = cwd.parent
    # theme_path = project_root_dir / "data" / "theme_words"
    # email_raw_path = project_root_dir / "output" / 'csv_file' / 'email_raw.csv'
    # facebook_url_info_path = project_root_dir / 'output' / 'csv_file' / 'facebook_urls_info.csv'
    # youtube_url_info_path = project_root_dir / 'output' / 'csv_file' / 'youtube_urls_info.csv'
    # log_file_path = project_root_dir / 'output' / 'log_file' / 'url_cleaning.log'
    # removed_email_path = project_root_dir / 'output' / 'csv_file' / 'removed_email_from_decrypteur.csv'
    # dataclean_pickle_path = project_root_dir / 'output' / 'pickle_obj' / 'dataClean'
    #
    # logging.basicConfig(filename=log_file_path, format="%(message)s", filemode='w', level=logging.INFO)
    #
    # email_raw_converters = {"attach_type": lambda x: x.strip("[]").replace("'", "").lower().strip().split(", "),
    #                         "urls": lambda x: x.strip("[]").replace("'", "").strip().split(', ')
    #                         }
    # email_df_raw = EmailDF.from_csv(csv_path=email_raw_path, converters=email_raw_converters)
    # email_df_raw.read_dict_theme_from_csv(theme_path)
    #
    # email_df = (email_df_raw.filt_by_date(start_date=(2020, 3, 1), end_date=(9999, 12, 30))
    #             .remove_email_by_from(logpath=removed_email_path)
    #             .clean_urls()
    #             .add_domain_column()
    #             .add_has_url_column()
    #             .add_text_column()
    #             .add_theme_column()
    #             .add_source_column())
    #
    # fb_yt_converter = {'theme': lambda x: x.strip("[]").replace("'", "").strip().split(', ')}
    # fb_df = UrlDF.from_csv(facebook_url_info_path, 1, fb_yt_converter, parse_dates=None)
    # yt_df = UrlDF.from_csv(youtube_url_info_path, 1, fb_yt_converter, parse_dates=None)