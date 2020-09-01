"""main.py: parsing mbox file to extract data for email analysis

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"
"""

import mailbox
import MboxParser
from EmailAnalyser import EmailAnalyser
from EmailDataCleaner import EmailDataCleaner
from FacebookScraper import FacebookScraper
from YoutubeScraper import YoutubeScraper
import TextCleaner
from tqdm import tqdm

import logging
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ --------------------LOAD DATA FROM MBOX-----------------------"""
    # path_mbox = pathlib.PureWindowsPath(os.getcwd()) / "data" / 'courriel_decrypteur_10_08_2020.mbox'
    # mbox = mailbox.mbox(path_mbox)
    # mboxParser = MboxParser.MboxParser(mbox)
    #
    # print('parsing raw email')
    # mboxParser.parse_email()
    # print()
    # print('exporting raw email to csv')
    # path_export_email_raw = pathlib.PureWindowsPath(os.getcwd()) / 'output' / 'csv_file'
    # mboxParser.export(path=path_export_email_raw, filename='email_raw')
    # print()
    # print('parsing and exporting email attachment')
    # mboxParser.export_attachment(path=None)
    # print()
    # print("done")
    """ --------------------    CLEAN DATA     -----------------------"""
    cwd = pathlib.PurePath(os.getcwd())
    project_root_dir = cwd.parent
    theme_path = project_root_dir / "data" / "theme_words"
    email_raw_path = project_root_dir / "output" / 'csv_file' / 'email_raw.csv'
    facebook_url_info_path = project_root_dir / 'output' / 'csv_file' / 'facebook_urls_info.csv'
    youtube_url_info_path = project_root_dir / 'output' / 'csv_file' / 'youtube_urls_info.csv'
    log_file_path = project_root_dir / 'output' / 'log_file' / 'url_cleaning_log.csv'
    removed_email_path = project_root_dir / 'output' / 'csv_file' / 'removed_email_from_decrypteur.csv'
    dataclean_pickle_path = project_root_dir / 'output' / 'pickle_obj' / 'dataClean'

    logging.basicConfig(filename=log_file_path, format="%(message)s", filemode='w', level=logging.INFO, encode='utf-8')

    #
    # logging.basicConfig(filename=project_root_dir/'output'/'log_file'/'test.log')
    # log1 = logging.getLogger('test.log')
    # log1.info('hello')

    dataCleaner = EmailDataCleaner.from_csv(filepath=email_raw_path)
    dataCleaner.filter_by_start_date(2020, 3, 1)
    dataCleaner.remove_email_from_decrypteur(logpath=removed_email_path)
    dataCleaner.clean_urls()
    dataCleaner.add_domain()
    dataCleaner.add_has_url()
    dataCleaner.parse_text()
    dataCleaner.parse_dict_theme_from_csv(theme_path)
    dataCleaner.add_theme()
    dataCleaner.add_source()
    fb_df = pd.read_csv(facebook_url_info_path, index_col=1,
                        converters={'theme': lambda x: x.strip("[]").replace("'", "").strip().split(', ')})
    yt_df = pd.read_csv(youtube_url_info_path, index_col=1,
                        converters={'theme': lambda x: x.strip("[]").replace("'", "").strip().split(', ')})
    fb_df.loc['theme'] = fb_df['theme'].replace('', None)
    yt_df.loc['theme'] = yt_df['theme'].replace('', None)

    log_fb = dataCleaner.add_theme_tag_from_url_scraper_data(fb_df, 'facebook')
    logb_yt = dataCleaner.add_theme_tag_from_url_scraper_data(yt_df, 'youtube')
    dataCleaner.save(dataclean_pickle_path)

    dataCleaner = EmailDataCleaner.from_pickle(dataclean_pickle_path)
    analyser = EmailAnalyser.from_EmailDataCleaner(dataCleaner)

    # print('parsing facebook urls')
    # facebook_urls = analyser.data.explode('urls')[analyser.data.domain.explode() == 'facebook']['urls'].value_counts()
    # data = []
    # for url in tqdm(facebook_urls.index):
    #     fs = FacebookScraper(url, analyser.theme)
    #     fs.parse_data()
    #     data.append(fs.export_as_list())
    # df_urls_facebook = pd.DataFrame(data, columns = ['url', 'title', 'published_date', 'comments', 'reactions', 'shares', 'views', 'theme'])
    # df_urls_facebook.to_csv(facebook_url_info_path)
    #
    #
    # print('parsing youtube urls')
    # youtube_urls = analyser.data.explode('urls')[analyser.data.domain.explode() == 'youtube']['urls'].value_counts()
    # data = []
    # for url in tqdm(youtube_urls.index):
    #     fs = YoutubeScraper(url, analyser.theme)
    #     fs.parse_data()
    #     data.append(fs.export_to_list())
    # df_urls_youtube = pd.DataFrame(data, columns = ['url', 'title', 'date_published', 'description', 'author', 'rating', 'views', 'theme'])
    # df_urls_youtube.to_csv(youtube_url_info_path)
