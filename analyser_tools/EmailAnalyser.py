import pandas as pd

import TextCleaner


def clean_corpus(corpus):
    corpus['text'] = corpus['text'].apply(lambda text: TextCleaner.clean_text_for_analysis(text))
    return corpus


class EmailAnalyser:

    def __init__(self, data=None, theme=None, N=None, corpus_per_email=None, dtm_per_email=None,
                 dtm_unique_per_email=None,
                 corpus_per_datetime=None, dtm_per_datetime=None, dtm_unique_per_datetime=None):
        self.data = data
        self.N = len(self.data) if self.data is not None else None
        self.theme = theme  # dict : keys = theme, values = word in theme

        self.corpus_per_email = self.set_corpus(by='email_id') if (
                    corpus_per_email is not None and self.data is not None) else None
        self.dtm_per_email = self.set_dtm(by='email_id') if (
                    dtm_per_email is not None and self.data is not None) else None
        self.dtm_unique_per_email = self.set_dtm_unique(by='email_id') if (
                    dtm_unique_per_email is not None and self.data is not None) else None

        self.corpus_per_datetime = self.set_corpus(by='datetime', sampling='D') if (
                    corpus_per_datetime is not None and self.data is not None) else None
        self.dtm_per_datetime = self.set_dtm(by='datetime', sampling='D') if (
                    dtm_per_datetime is not None and self.data is not None) else None
        self.dtm_unique_per_datetime = self.set_dtm_unique(by='datetime') if (
                    dtm_unique_per_datetime is not None and self.data is not None) else None

    def plot_email_per_word_in_theme(self, pourc=True, n_col=5, plot_name='word_in_theme', without_theme=[], save=False,
                                     show=True):
        """return a figure with 3x5 subplots of bar plot. y axis is shared.
        xaxis is word in theme
        each subplot is a theme
        y axis is either % or count of email with the word in x axis
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        n_row = math.ceil(len(self.theme.keys()) / n_col)  # nomber of row to show all plot with n_col
        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 10), facecolor='w', edgecolor='k', sharey=True)
        fig.subplots_adjust(hspace=1, wspace=.001)

        axs = axs.ravel()
        if pourc:
            series_word_in_theme = (self.get_email_per_word_in_theme() / self.N) * 100
        else:
            series_word_in_theme = self.get_email_per_word_in_theme()
        for i, theme in enumerate(self.theme.keys()):
            s = series_word_in_theme.loc[theme]  # word and count value
            words = s.index
            values = s.values
            if theme in without_theme:
                values = np.zeros(len(values))
                axs[i].bar(words, values)

            axs[i].bar(words, values)
            axs[i].set_title(theme)

            for tick in axs[i].get_xticklabels():
                tick.set_rotation(90)

        if save:
            plt.savefig(f'{plot_name}.png', dpi=300)
        if show:
            plt.show()

    def get_proportion_email_per_theme(self):
        count_per_day = self.data.explode('theme').groupby(['year', 'month', 'day'])['theme'].value_counts().unstack()
        count_per_day[count_per_day.isnull()] = 0
        for index in count_per_day.index:
            year, month, day = index[0], index[1], index[2]
            num_email_that_day = self.get_freq_email_per_day().loc[f'{year}-{month}-{day}']
            count_per_day.loc[index] /= num_email_that_day

        return count_per_day

    def plot_theme_per_day(self, pourc=True, n_col=5, save=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        n_row = math.ceil(len(self.theme.keys()) / 5)  # nomber of row to show all plot with n_col
        fig, axs = plt.subplots(n_row, 5, figsize=(15, 10), facecolor='w', edgecolor='k', sharey=True)
        fig.subplots_adjust(hspace=1, wspace=.001)

        axs = axs.ravel()
        if pourc:
            em_per_d = self.get_freq_email_per_day()
            theme_per_day = self.get_theme_freq_per_day().apply(lambda x: (x / em_per_d) * 100)
        else:
            theme_per_day = self.get_freq_email_per_day()

        for i, theme in enumerate(self.theme.keys()):
            s = theme_per_day[theme]  # word and count value
            days = s.index
            values = s.values

            axs[i].plot(days, values)
            axs[i].set_title(theme)

            for tick in axs[i].get_xticklabels():
                tick.set_rotation(90)

        if save:
            plt.savefig('theme_per_day.png', dpi=300)

    def plot_email_per_theme(self, prop=False, plot_name='email_per_theme', save=False, show=True):
        """un seul bar plot avec theme sur x et nombre de courriel sur y"""

        import matplotlib.pyplot as plt

        s = self.get_email_per_theme()
        index = self.theme.keys()

        values = (s.values / self.N) if prop else s.values
        label = "pourcentage" if prop else 'nombre de courriel'

        f = plt.figure(figsize=(10, 10))

        plt.tight_layout()
        plt.title(plot_name)

        plt.bar(index, values, label=label)
        plt.xticks(rotation=90)

        plt.legend(loc='best')
        if show:
            plt.show()
        if save:
            plt.savefig(f'{plot_name}.png', dpi=300)

    def get_email_with_len_theme(self, num_theme=0, cond='=='):
        """retourne le dataframe avec les emails qui ont un certain nombre de theme de taggé"""
        # TODO seulement texter pour ==, tester le reste
        if cond == '<':
            return self.data[self.data['theme'].apply(lambda x: len(x) < num_theme)]
        elif cond == '>':
            return self.data[self.data['theme'].apply(lambda x: len(x) > num_theme)]
        elif cond == '<=':
            return self.data[self.data['theme'].apply(lambda x: len(x) <= num_theme)]
        elif cond == '>=':
            return self.data[self.data['theme'].apply(lambda x: len(x) >= num_theme)]
        else:
            return self.data[self.data['theme'].apply(lambda x: len(x) == num_theme)]

    def get_freq_email_per_day(self, sampling='D'):
        return self.data.set_index('datetime').resample(sampling).count()['email_id']

    def get_theme_per_sampling(self, sampling='D'):
        return self.data.set_index('datetime').reseample(sampling).agg({''})

    def get_email_per_theme_partition(self):
        """trouve la parition si elle n'existe pas de tous les themes O(2^n). Permet de trouver les combinaisons
        de 1,2,3,4,... theme avec le plus de courriel."""
        import EmailTheme
        data = []
        index = []
        partition = None
        try:
            themeObj = EmailTheme.from_pickle("theme_partition.pickle_obj")
            partition = themeObj.partition
        except FileNotFoundError:
            print("partition does not exist")
        finally:
            if partition is None:
                print('creating partition')
                themeObj = EmailTheme.EmailTheme.from_dict_theme(self.theme)
                themeObj.set_theme_partitions(max_partition=None)
                partition = themeObj.partition

        for p in partition:
            index.append(tuple(p))
            data.append(len(self.get_email_with_theme(with_theme=p)))

        return pd.Series(data, index=index).sort_values(ascending=False)

    def get_num_email_with_url(self, with_urls=[]):
        return len(self.get_email_with_url(with_urls=with_urls))

    def get_num_email_with_attachment(self):

        return

        pass

    def get_freq_email_for_each_attach_type(self):
        pass

    def get_num_url(self):
        return self.data.urls.apply(lambda x: len(x)).sum()

    def get_email_with_theme_partition(self):
        return self.data[['email_id', 'theme']]['theme'].value_counts()

    def get_word_freq_per_day(self):
        pass

    def get_theme_freq_per_day(self):
        """assemble les themes par jour et retour un dataframe avec comme index les journées,
        comme colonnes les themes et comme valeur la fréquence d'apparition de chaque thème dans la journeé"""
        df_per_day = self.data.set_index(pd.DatetimeIndex(self.data['datetime'])).resample('D').agg({'theme': sum})
        data = {theme: [] for theme in self.theme.keys()}
        for d in df_per_day.index:
            for theme in self.theme.keys():
                if df_per_day.loc[d, 'theme'] == 0:
                    count = 0
                else:
                    count = df_per_day.loc[d, 'theme'].count(theme)
                data[theme].append(count)
        return pd.DataFrame(data, index=df_per_day.index)

    def get_most_common_url(self, ascending=False):
        return self.data.urls.explode().value_counts().drop('').sort_values(ascending=ascending)

    def resample_per_datetime(self, colnames, function, sampling='D'):
        return self.data.set_index('datetime').resample(sampling).agg({colnames: function})

    def most_common_word(self):
        return self.dtm.sum().sort_values(ascending=False)

    def most_common_unique_word(self):
        return self.dtm_unique.sum().sort_values(ascending=False)

    def get_email_per_word_in_theme(self):
        """:return multiindex level 0 = theme, level 1 = word in theme, value = num_email with word in theme"""
        multiindex = self.multiindex_from_theme()
        data = {theme_word: self.get_email_with_value_in_column('text', with_value=TextCleaner.get_word_variation(
            theme_word[1]), condition='or').shape[0] for theme_word in multiindex}
        return pd.Series(data)

    def multiindex_from_theme(self):
        index = []
        for theme in self.theme.keys():
            for word in self.theme[theme]:
                index.append((theme, word))
        return pd.MultiIndex.from_tuples(index)

    def get_email_per_theme(self):
        # dict comprehensin {key: value for (key, value) in iterable}
        print('fetching count email per theme')
        data = {theme: self.get_email_with_value_in_column('theme', with_value=[theme]).shape[0] for theme in
                self.theme.keys()}
        return pd.Series(data)

    def get_email_with_value_in_column(self, columns, with_value=[], condition="and", from_df=None):
        """retourne un df avec seulement les courriels contenants les thèmes demandés.
        (possibilité de restreindre à un des thèmes ou tous les thèmes"""
        condition_and = lambda values: all(v in values for v in with_value)  # si tous les themes demander
        condition_or = lambda values: any(v in values for v in with_value)  # si au moins un des thèmes demander
        condition = condition_and if condition == 'and' else condition_or  # choisir une des deux conditions
        if from_df is not None:
            df_output = from_df[from_df[columns].apply(condition)]
        else:
            df_output = self.data[self.data[columns].apply(condition)]
        return df_output

    def get_corpus(self, by='email_id', sampling='D'):
        if by == 'email_id':
            return self.corpus_per_email

        elif by == 'datetime':
            return self.corpus_per_datetime

    def set_corpus(self, by='email_id', sampling='D'):
        if by == 'email_id':
            corpus = self.data[['email_id', 'text']].set_index('email_id')
            self.corpus_per_email = corpus

        elif by == 'datetime':
            corpus = self.data.set_index('datetime').resample(sampling).agg({'text': ' '.join})
            self.corpus_per_datetime = corpus

    def get_dtm(self, by='email_id'):
        if by == 'email_id':
            return self.dtm_per_email
        elif by == 'datetime':
            return self.dtm_per_datetime

    def set_dtm(self, by='email_id', sampling='D'):
        from stop_words import get_stop_words
        from sklearn.feature_extraction import text
        from sklearn.feature_extraction.text import CountVectorizer

        french_stop_words = get_stop_words('french')
        english_stop_words = list(text.ENGLISH_STOP_WORDS)
        my_stop_words = french_stop_words + english_stop_words
        if by == 'email_id':
            corpus = self.corpus_per_email
            corpus = clean_corpus(corpus)
        elif by == 'datetime':
            self.set_corpus(by, sampling)
            corpus = self.corpus_per_datetime
            corpus = clean_corpus(corpus)

        # print(french_stop_words)
        cv = CountVectorizer(stop_words=my_stop_words)
        data_cv = cv.fit_transform(corpus.text)
        dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        dtm.index = corpus.index

        if by == 'email_id':
            self.dtm_per_email = dtm
        elif by == 'datetime':
            self.dtm_per_datetime = dtm

    def get_unique_dtm(self, by='email_id'):
        if by == 'email_id':
            return self.dtm_unique_per_email

        if by == 'datetime':
            return self.dtm_unique_per_datetime

    def set_dtm_unique(self, by='email_id'):
        if by == 'email_id':
            dtm_unique = self.dtm_per_email.copy()
            dtm_unique[dtm_unique >= 1] = 1
            self.dtm_unique_per_email = dtm_unique

        if by == 'datetime':
            dtm_unique = self.dtm_per_datetime.copy()
            dtm_unique[dtm_unique >= 1] = 1
            self.dtm_unique_per_datetime = dtm_unique

    def set_prop_email_using_each_word_per_day(self):
        print('creating dtm with pourc email with word per day')
        email_per_day = self.get_freq_email_per_day()
        self.prop_with_word_per_day = self.dtm_unique_per_datetime.apply(lambda x: (x / email_per_day) * 100)

    def get_prop_email_word_per_word(self):
        return self.prop_with_word_per_day

    @classmethod
    def from_EmailDataCleaner(cls, dataCleaner):
        return EmailAnalyser(data=dataCleaner.data, theme=dataCleaner.theme)
