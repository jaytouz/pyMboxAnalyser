"""MboxParser.py: access mbox object from mailbox module and iterate through
    all emails to extract data one by one using EmailParser

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"

"""
import mailbox
from pathlib import PureWindowsPath

import pandas as pd
import os
import EmailParser
import logging


class MboxParser:
    def __init__(self, takeout_mbox):
        self._mbox = takeout_mbox
        self.data = None
        self.cwd = os.getcwd()

    def __len__(self):
        return len(self._mbox)

    def parse_email(self, with_url_status=False):
        from tqdm import tqdm
        header = ['email_id', 'from', 'to', 'ip', 'datetime', 'day', 'month', 'year', 'weekofyear',
                  'has_attach', 'attach_type', 'num_urls', 'urls', 'domain', 'subject', 'body', 'num_words']

        data = dict.fromkeys(header)
        # cannot create with (header,[]) because it makes all key point to the same reference.
        for k, _ in data.items():
            data[k] = []

        for m, em_id in tqdm(zip(self._mbox, list(range(len(self._mbox)))), total=len(self._mbox)):
            em_parser = EmailParser.EmailParser(m, em_id, with_url_status)
            em_parser.parse_all_data()
            _from = em_parser.get('from')
            _to = em_parser.get('to')
            _ip = em_parser.get('ip')
            _date = em_parser.get('date')

            data['email_id'].append(em_id)
            data['from'].append(_from)
            data['to'].append(_to)
            data['datetime'].append(str(_date))
            data['day'].append(_date.day)
            data['month'].append(_date.month)
            data['year'].append(_date.year)
            data['weekofyear'].append(_date.isocalendar()[1])
            data['ip'].append(_ip)
            data['has_attach'].append(em_parser.get("has_attachment"))
            data['attach_type'].append(em_parser.get("attachment_type"))
            data['num_urls'].append(len(em_parser.get("urls")))
            data['urls'].append(em_parser.get('urls'))
            data['domain'].append(em_parser.get('urls_domain'))
            data['subject'].append(em_parser.get('subject'))
            data['body'].append(em_parser.get('body'))
            data['num_words'].append(em_parser.get('num_words'))

        self.data = data

    def export_attachment(self, path=None):
        """
        :param path: string of pure windows path from cwd (Ex: '\\output\\email_attachments'
        :return:
        """
        from tqdm import tqdm

        # TODO  make it work in MacOS
        if path is None:
            path = PureWindowsPath(os.getcwd() + '\\output\\email_attachments')
        else:
            path = PureWindowsPath(os.getcwd() + path)

        if not os.path.exists(path):
            os.makedirs(path)

        for m, em_id in tqdm(zip(self._mbox, list(range(len(self._mbox)))), total=len(self._mbox)):
            em_parser = EmailParser.EmailParser(m, em_id, with_url_status=False)
            em_parser.parse_all_data()
            has_attach = em_parser.get('has_attachment')
            if has_attach:
                attachment_type = em_parser.get('attachment_type')
                attachments = em_parser.get('attachments')

                for content_type, raw_content, attach_id in zip(attachment_type, attachments,
                                                                list(range(len(attachments)))):
                    file_ext = EmailParser.get_file_ext(content_type)
                    EmailParser.save_attachment(path, raw_content, em_id, attach_id, file_ext)

    def export(self, path, file_ext='csv', attrs='data', sep=','):
        """
        :param path:
        :param filename:
        :param file_ext: file extension
        :param attrs: data is the default attrs, it contains all the attribute that have been parsed in parse_email function
        :param sep: seperator
        """
        data_dict = self.__getattribute__(attrs.strip().lower())  # data to save in dictionnary format

        df = pd.DataFrame(data_dict)

        if not os.path.exists(path):
            os.makedirs(path)

        if file_ext == 'csv':
            df.to_csv(path, index = False, sep=sep, encoding='utf-8')


if __name__ == "__main__":
    LOG_FILENAME = 'email_parser.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    logging.debug('-----------Start----------')
    mbox = mailbox.mbox('courriel_decrypteur_19_06_2020.mbox')  # iterable
    mboxParser = MboxParser(mbox)

    print('parsing raw email')
    mboxParser.parse_email()
    print('exporting raw email to csv')
    mboxParser.export(path=None, filename='email_raw', file_ext='csv')

    print('parsing and exporting email attachment')
    mboxParser.export_attachment(path=None)
