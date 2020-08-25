"""EmailParser.py: read an email using mailbox module, format information and store data

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"
"""

# importation
import os
from pathlib import PureWindowsPath

import numpy as np
import mailbox
import urllib3

import logging


# Method static

def html_text_body(html_obj):
    from bs4 import BeautifulSoup
    full_text = []
    text = html_obj.get_payload(decode=True)
    soup = BeautifulSoup(text, 'html.parser')
    parts = soup.find_all(dir="auto")

    if len(parts) == 0:
        # cas particulier... je ne crois pas qu'il y en a d'autre.
        parts = soup.find_all('p')

    for part in parts:
        full_text.append(part.text)

    return " ".join(full_text)


def parse_email_from_text(text):
    start = 0
    end = len(text) - 1
    if '<' in text:
        start = text.find('<')
    if '>' in text:
        end = text.find('>')
    email = text[start + 1:end]
    return email


def body_string_correction(text, replacement=None):
    """correction de certain string manuellement lorsque néessaire"""
    if replacement is None:
        replacement = {"=E9": "é", "=E8": "è", "=C3=A9": "é",
                       "=C3=A8": "è", "=C3=A0": "à", "_": " ",
                       "=C7": "ç"}
    for k, v in zip(replacement.keys(), replacement.values()):
        text = text.replace(k, v)
    return text


def find_ext(content_ext):
    doc_types_ext = {"vnd.openxmlformats-officedocument.spreadsheetml.sheet": 'xlsx',
                     "vnd.openxmlformats-officedocument.wordprocessingml.document": 'docx',
                     'mpeg': 'mp4',
                     "x-m4a": 'mp4',
                     "jpeg": "jpeg",
                     "jpg": "jpeg",
                     "png": "png",
                     "gif": "gif",
                     "html": "html",
                     "tiff": "tiff",
                     "mp4": "mp4",
                     "plain": "txt",
                     "quicktime": 'mp4',
                     "pdf": "pdf"
                     }

    file_ext = None
    if content_ext in doc_types_ext:
        file_ext = doc_types_ext[content_ext]
    return file_ext


def save_attachment(path, raw_content, email_id, attach_id, file_ext, bytes_format='utf-8', save=True,
                    get_output=False):
    """fonction to decode raw byte and save it to appropriate extention to open in file browser"""
    import base64
    output = raw_content
    if file_ext is None:
        """used in MboxParser"""
        return
    if file_ext != 'html' and file_ext != 'txt':
        if type(output) != bytes:
            output = base64.decodebytes(output.encode(bytes_format))
    if save:
        with open(path / f"{email_id}_{attach_id}.{file_ext}", 'wb') as f:
            # because path is Path Object from pathlib '/' is used to concatenate
            f.write(output)
    if get_output:
        return output


def get_file_ext(content_type):
    content_ext = content_type.split('/')[1].lower()
    file_ext = find_ext(content_ext)
    return file_ext


class EmailParser:
    def __init__(self, email, email_id, with_url_status):
        self.email = email

        self.email_id = email_id
        self._to = None
        self._from = None
        self._subject = None
        self._date = None
        self._ip = None
        self._charset = None
        self._has_attachment = None
        self._attachment_type = None  # TODO est-ce que ca devrait etre une liste vide?
        self._attachments = None
        self._num_words = None
        self._urls = None
        self._urls_domain = None
        self._urls_status = None
        self._body = None

        self.with_url_statut = with_url_status

    def set_to(self, val):
        self._to = val

    def set_from(self, val):
        self._from = val

    def set_subject(self, val):
        self._subject = val

    def set_date(self, val):
        self._date = val

    def set_charset(self, val):
        self._charset = val

    def set_ip(self, val):
        self._ip = val

    def set_has_attachment(self, val):
        self._has_attachment = val

    def set_attachment_type(self, val):
        self._attachment_type = val

    def set_body(self, val):
        self._body = val

    def set_urls(self, val):
        self._urls = val

    def set_urls_domain(self, val):
        self._urls_domain = val

    def set_urls_status(self, val):
        self._urls_status = val

    def set_num_words(self, val):
        self._num_words = val

    def parse_all_data(self):
        self.parse_to()
        self.parse_from()
        self.parse_subject()
        self.parse_date()
        self.parse_charset()
        self.parse_ip()
        self.parse_attachment()
        self.parse_body()
        if self._body is not None:
            self.parse_urls()
            self.parse_urls_domain()
            if self.with_url_statut:
                self.parse_url_status()
            self.parse_num_words()

    def parse_num_words(self):
        num_words = len(self.text_to_list('body')) + len(self.text_to_list('subject'))
        self.set_num_words(num_words)

    def parse_to(self):
        to = self.email["TO"]
        if to is None:
            to = self.email['cc']
            if to is None:
                to = self.email['X-BeenThere']
        all_to = to.split(',')
        if len(all_to) > 1:
            temp = []
            for t in all_to:
                if '<' in t and '>' in t:
                    temp.append(parse_email_from_text(t))
                else:
                    temp.append(t)
            to = ", ".join(temp)
        else:
            if '<' in to and '>' in to:
                to = parse_email_from_text(to)  # keep only inside <kajskfd.fsfa@gmail.com>

        self.set_to(to)

    def parse_from(self):
        _from = self.email["FROM"]
        all_from = _from.split(',')
        if len(all_from) > 1:
            temp = []
            for f in all_from:
                if '<' in f and '>' in f:
                    temp.append(parse_email_from_text(f))
                else:
                    temp.append(f)
            _from = ", ".join(temp)
        else:
            if '<' in _from and '>' in _from:
                _from = parse_email_from_text(_from)  # keep only inside <kajskfd.fsfa@gmail.com>

        if ',' in _from:
            _from = _from.split(',')[1].strip()

        self.set_from(_from)

    def parse_subject(self):
        subject = self.email['subject']
        try:
            if (subject is not None and subject != '') and subject.strip()[0] == '=':
                """if format looks like '=?UTF-8?Q?publication_=C3=A0_v=C3=A9rifier_SVP_Merci!?=;"""
                subs = subject.split('?')
                subject = subs[-2]  # subject = publication_=C3=A0_v=C3=A9rifier_SVP_Merci!
                self._subject = body_string_correction(subject)
        except AttributeError:
            """happens when subject is a Header object."""
            # TODO should be an if statement not in catch.
            subject = str(self.email['subject'])

        self.set_subject(subject)

    def parse_date(self):
        from dateutil import parser
        try:
            dt = parser.parse(self.email['date'])
        except TypeError:
            # one email return a Header object..
            date_str = str(self.email['date']).split('(')[0].strip()
            dt = parser.parse(date_str)

        self.set_date(dt)

    def parse_ip(self):
        ip = np.nan
        if self.email['x-originating-ip'] is not None:
            ip = self.email["x-originating-ip"]

        self.set_ip(ip)

    def parse_attachment_type(self):
        types = []
        for part in self.email.walk():
            if part["Content-Disposition"] is not None:
                _type = part['Content-Type'].split(";")[0]
                types.append(_type)

        self.set_attachment_type(types)

    def parse_has_attachment(self):
        has_attachment = False
        for part in self.email.walk():
            if part["Content-Disposition"] is not None:
                has_attachment = True
        self.set_has_attachment(has_attachment)

    def parse_attachment(self):
        attachments = []
        self.parse_has_attachment()
        if self._has_attachment:
            self.parse_attachment_type()
            attach_id = 0
            for part in self.email.walk():
                if part['content-disposition'] is not None:
                    content_type = part['content-type']  # text/html; CHARSET=ISO-8859-1; name=Horreur!.html
                    _type = content_type.split(';')[0].split('/')[0].lower()
                    if _type.lower() == 'null':
                        # certain courriel on du content-type sans type. impossible a ouvrir
                        continue
                    content_ext = content_type.split(';')[0].split('/')[1].lower()
                    file_ext = find_ext(content_ext)

                    if file_ext is None:
                        # skip les fichiers qui ne sont pas dans doc_type_ext
                        continue

                    raw_content = part.get_payload(decode=True)
                    output_path = PureWindowsPath(os.getcwd() + '\\output\\attachment_email_decrypteur')
                    output = save_attachment(output_path, raw_content, self.email, attach_id, file_ext, save=False,
                                             get_output=True)
                    attachments.append(output)
                    attach_id += 1

        self._attachments = attachments

    def parse_charset(self):
        """return the first charset from the first text/plain"""
        charset = "nan"
        try:
            for part in self.email.walk():
                # looking for a string like this in part['content-type'] "text/plain; charset="UTF-8""
                content = part['content-type'].split(';')[0]
                if content.lower() == 'text/plain' and charset == 'nan':
                    content_param = part['content-type'].split(';')  # content-type, format, charset
                    for p in content_param:
                        if "charset" in p.lower():
                            charset = p.split('=')[1].replace('"', '')

            if charset == 'nan':
                # look in htlm text
                for part in self.email.walk():
                    content = part['content-type'].split(';')[0]
                    if content.lower() == 'text/html':
                        content_param = part['content-type'].split(';')  # content-type, format, charset
                        for p in content_param:
                            if "charset" in p.lower():
                                charset = p.split('=')[1].replace('"', '')
        except AttributeError:
            # TODO be more precise with the catch.
            # catching errors because there is no body, therefore no charset.
            # print(e)
            pass
        if charset.lower() in ['us-ascii']:
            # because every us-ascii is 8bit based encoding and every character map to the same thing in utf-8.
            charset = 'utf-8'
        self._charset = charset

    def parse_body(self):
        self._body = ''
        messages = []
        if self.email.get_payload() != '':
            # s'il y a un body

            for part in self.email.walk():
                content = part['content-type']
                if content != '' and content is not None:
                    content = content.split(';')[0].lower()
                if content == 'text/plain':
                    text = ''
                    try:
                        text = part.get_payload(decode=True).decode(self._charset)
                    except UnicodeDecodeError as e:
                        logging.debug(str(self.email_id) + " " + str(e))
                    finally:
                        messages.append(text)

            if len(messages) == 0:
                # no text/plain, look for text/html
                for part in self.email.walk():
                    content = part['content-type']
                    if content != '' and content is not None:
                        content = content.split(';')[0].lower()
                    if content == 'text/html':
                        logging.log(20, str(self.email_id) + " got content from html")
                        text = html_text_body(part)
                        messages.append(text)

            body = " ".join(messages)
            body = body.encode('utf-8', errors='ignore').decode('utf-8')  # making sure it's utf-8 with no unicode
            if len(body) > 5000:
                body = body[:5000]
            self._body = body

    def parse_urls(self):
        """
        Extract url from body and subject
        :return:
        """
        from copy import deepcopy
        urls = []
        body = deepcopy(self._body)
        subject = str(self._subject)
        words_list = body.replace("\n", " ").replace(">", " ").replace("<", " ").strip().split(' ')

        for w in words_list:
            if "http".lower() in w.lower():
                urls.append(w)
        if "http".lower() in subject.lower():
            url_in_subject = self.parse_url_from_subject()
            urls.append(url_in_subject)

        self._urls = urls

        if len(urls) > 0:
            self.clean_url()

    def parse_urls_domain(self):
        urls_domain = []
        for url in self._urls:
            # EX : "https://m.youtube.com/watch?v=nFPeN17PVU8&feature=youtu.be" => m.youtube.com
            try:
                domain = url.split('//')[1].split('/')[0].lower()  # m.youtube.com
            except IndexError:
                domain = 'error'

            # test for youtube
            if 'youtu' in domain:
                domain = 'youtube'
            elif 'facebook' in domain:
                domain = 'facebook'
            else:
                domain = 'other'
            urls_domain.append(domain)

        self._urls_domain = urls_domain

    def parse_url_status(self):
        http = urllib3.PoolManager()

        urls_status = []
        for url in self._urls:
            status = 0
            try:
                r = http.request('GET', url,
                                 retries=urllib3.Retry(raise_on_redirect=False), timeout=3)
                status = r.status
            except urllib3.exceptions.HTTPError:
                status = 'error'
            finally:
                urls_status.append(status)

        self._urls_status = urls_status

    def text_to_list(self, attr):
        """retire les sauts de ligne les >, ',', et les '.'"""
        text = self.get(attr)
        words_list = []
        if text is not None and text != '':  # TODO voir si pour du NPL il faut garder point et virgule.
            words_list = text.replace("\n", " ").replace(">", " ").replace(',', ' ').replace('.', ' ').strip().split()
        return words_list

    def parse_url_from_subject(self):
        url = self._subject  # if nothing better, keep subject
        split_space = self._subject.split(' ')
        for w in split_space:
            if 'http' in w.lower():
                url = w
        return url

    def get(self, attr):
        return self.__getattribute__("_{}".format(attr.strip().lower()))

    def clean_url(self):
        for url, url_idx in zip(self._urls, list(range(len(self._urls)))):
            if len(url) < 12:
                # minimum pour un url valid avec un domaine de 1 char et non securisé (http)
                self._urls.remove(url)
                continue
            start_http = url.find('http')
            if start_http > 0:
                # crop tout ce qui est avant http
                url = url[start_http:]

            last_char = url[-1]
            if last_char in [')', ']']:
                url = url[:-1]  # coupe le dernier charatère

            # save change
            self._urls[url_idx] = url


if __name__ == "__main__":
    #### FOR TEST
    mbox = mailbox.mbox('courriel_decrypteur_19_06_2020.mbox')  # iterable
    i = 0
    target = 4499
    for m in mbox:
        if i == target:
            break
        i += 1
    em_parser = EmailParser(m, False)
    em_parser.parse_all_data()
