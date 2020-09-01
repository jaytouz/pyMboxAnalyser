"""UrlCleaner.py: Class pour standardiser les ULRS ou supprimés ceux qui sont du spams.

__author__      = "Jérémie Tousignant"
__copyright__   = "Copyright 2020, Radio-Canada"

"""

import re
import logging


def get_domain(url):
    try:
        raw_domain = url.split("//")[1].split('/')[0]
    except IndexError:
        return "NaN"
    num_dot = raw_domain.count('.')
    if num_dot == 1:
        domain = raw_domain.split('.')[0]
    else:
        if num_dot == 0:
            domain = raw_domain
        else:
            domain = raw_domain.split('.')[1]
    if domain == 'youtu':
        domain = 'youtube'

    if domain == 'co':
        # si on arrive la c'est que l'url est de la forme https://domain.co.xyz ou https://www.domain.co.xyz
        domain = raw_domain.split('.co')[0]

    return domain


def get_count_domain(url):
    domains = {}
    for u in url:
        if u != '':
            dom = get_domain(u)
            if dom not in domains.keys():
                domains[dom] = 1
            else:
                domains[dom] += 1
    return domains


def find_youtube_video_id(url):
    if 'youtu.be' in url[:30]:  # pour s'assurer qu'on regarde la facon que le domain est ecrit
        search_res = re.findall("(?<=youtu.be/)[a-zA-Z0-9-_]+", url)
    else:
        search_res = re.findall("(?<=v=)[a-zA-Z0-9-_]+", url)

    if len(search_res) == 0:
        video_id = "not_a_video"
    else:
        video_id = search_res[0][:11]  # il ne peut y avoir plus de 11 caractères, sinon c'est une erreur

    return video_id


def find_facebook_page_id(url):
    search_res = re.findall("[\d]{15,17}", url)
    page_id = search_res[0] if len(search_res) >= 1 else None
    return page_id


class UrlCleaner:
    undesirable_domain = ['microsoft', 'can01', 'safelinks', 'mailtrack', 'avast', 'aka', 'avg', 'symantec', 'xx']

    def __init__(self, series_of_list):
        self.urls = series_of_list

    def remove_undesirable_domain(self):
        """retire les urls avec un domaine de la liste undesirable_domain"""
        urls_removed = []
        format_log = "{email_id},{url_removed}"
        spam_url_log = logging.getLogger('spam_url_log')
        for email_id, list_u in (zip(self.urls.index, self.urls)):
            if len(list_u) > 1 or list_u[0] != '':
                # s'il y a au moins un url a verifier
                for url_id, url in enumerate(list_u):
                    if url != '' and get_domain(url) in UrlCleaner.undesirable_domain:
                        self.urls[email_id][url_id] = ''
                        log_message = "spam_url_log," + format_log.format(email_id=email_id, url_removed=url)
                        spam_url_log.info(log_message)
                        urls_removed.append((email_id, url))

    def remove_repetition_of_domain(self, keep=5):
        format_log = "{email_id},{url_removed}"
        rep_url_log = logging.getLogger('rep_url_log')
        for email_id, list_u in (zip(self.urls.index, self.urls)):
            if len(list_u) > 1 or list_u[0] != '':
                # s'il y a au moins un url a verifier
                # trouver le domaine et le nombre de fois qu'il revient
                dom_count = get_count_domain(list_u)
                # initialiser le compteur pour s'assurer qu'il reste au moins 5 urls avec le domaine a supprimer.
                dom_extra_to_remove = {dom: count - keep for dom, count in zip(dom_count.keys(), dom_count.values()) if
                                       count > 5}

            for url_id, url in enumerate(list_u):
                if url != '':
                    dom = get_domain(url)
                    # s'il s'agit d'un domaine avec plus de 5 et qu'il y en a encore d'extra, supprimer en ordre d'apparition.
                    if dom_count[dom] > 5 and dom_extra_to_remove[dom] > 0:
                        self.urls[email_id][url_id] = ''
                        log_message = "rep_url_removed," + format_log.format(email_id=email_id, url_removed=url)
                        rep_url_log.info(log_message)
                        dom_extra_to_remove[dom] -= 1

    def standardise_youtube_url(self):
        format_log = "{old_url},{new_url}"
        y_logger = logging.getLogger('yt_url_modif')
        corrected = []
        for email_id, list_u in (zip(self.urls.index, self.urls)):
            if len(list_u) > 1 or list_u[0] != '':
                # s'il y a au moins un url a verifier
                for url_id, url in enumerate(list_u):
                    if url != '':
                        dom = get_domain(url)
                        # s'il s'agit d'un domaine avec plus de 5 et qu'il y en a encore d'extra, supprimer en ordre d'apparition.
                        if dom == 'youtube':
                            video_id = find_youtube_video_id(url)
                            if video_id != 'not_a_video':
                                new_url = f"https://www.youtube.com/watch?v={video_id}"
                                self.urls[email_id][url_id] = new_url
                                log_message = "yt_url_modif," + format_log.format(old_url=url, new_url=new_url)
                                y_logger.info(log_message)
                                # print(email_id, " ---", video_id," --- ", url, " --- ", new_url)

    def standardise_facebook_page_id(self):
        format_log = "{old_url},{new_url}"
        f_logger = logging.getLogger('fb_url_modif')
        for email_id, list_u in (zip(self.urls.index, self.urls)):
            if len(list_u) > 1 or list_u[0] != '':
                # s'il y a au moins un url a verifier
                for url_id, url in enumerate(list_u):
                    if url != '':
                        dom = get_domain(url)
                        if dom == 'facebook':
                            page_id = find_facebook_page_id(url)
                            if page_id is not None:
                                new_url = f"https://www.facebook.com/{page_id}/"
                                self.urls[email_id][url_id] = new_url
                                log_message = "fb_url_modif," + format_log.format(old_url=url, new_url=new_url)
                                f_logger.info(log_message)
                                # print(email_id, " ---", video id," --- ", url, " --- ", new_url)

        # logging removed urls
