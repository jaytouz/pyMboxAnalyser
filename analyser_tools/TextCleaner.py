def encoding_correction(text):
    """Decoding error to replace manually"""
    to_replace = {"=E9": "é", "=E8": "è", "=C3=A9": "é", "=C3=89": 'é',
                  "=C3=A8": "è", "=C3=A0": "à", "=C7": "ç", "": "'"}
    for (key, value) in zip(to_replace.keys(), to_replace.values()):
        text = text.replace(key, value)

    return text


def remove_extra_space(text):
    import re
    text = re.sub('\\n', ' ', text)
    text = re.sub('\\t', ' ', text)
    text = re.sub('\s{2,}', ' ', text)  # if two or more whitespace, replace with one
    return text


def remove_non_text_character(text):
    import re
    text = re.sub('[«»]', ' ', text)
    text = re.sub('[<>]', ' ', text)
    text = re.sub('[()]', ' ', text)
    text = re.sub('[//]', ' ', text)
    text = re.sub('[/]', ' ', text)
    text = re.sub("\\'", "'", text)  # corriger les apostrophes \' => '
    text = re.sub("-", ' ', text)
    text = re.sub("[!?:@.'’,_=]", ' ', text)  # remove punctuation
    text = re.sub('\s{2,}', ' ', text)  # if two or more whitespace, replace with one
    return text


def remove_word_with_only_number(text):
    import re
    text = re.sub('\s\d{5,}\s', ' ', text)
    return text


def remove_single_letter(text):
    import re
    text = re.sub('\s\w{1}\s', ' ', text)
    text = re.sub('^\w{1}\s', ' ', text)
    text = re.sub('\s\w{1}$', ' ', text)
    text = text.strip()

    return text


def remove_emojis(text):
    """https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python"""
    import emoji
    allchars = [str for str in text.encode().decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join(
        [str for str in text.encode().decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text


def remove_non_roman_character(text):
    import re
    text = re.sub('\S*[一-拿]+\S*', ' ', text)  # chinese words
    text = re.sub('\S*[一-拿]+\S*', ' ', text)
    text = re.sub(
        '[儘快儘量冠狀分鐘受到口罩只是只能咳嗽喉嚨喉痛喝多喝水因為因爲大大大陸如果存活完成對於就能市場幾天感染方法時候有效每個每天沒有洗手流行消滅減少準確漱口為麼然後王叔病毒病菌簡單綜合翻译肺炎肺部藥品訊息購買轉達這個過去醫師銷毀開始開水非常預防]',
        '', text)
    text = re.sub('\S*[α-ωΑ-Ω]+\S*', ' ', text)  # GREEK words
    text = re.sub('[ء-ي]+', '', text)  # Aracbic character
    text = re.sub('ھ', ' ', text)  # special case
    text = re.sub('\s{2,}', ' ', text)  # if two or more whitespace, replace with one
    return text


def remove_url(text):
    import re
    text = re.sub('https?://\S+', '', text)  # retirer les urls
    text = re.sub('\s{2,}', ' ', text)  # if two or more whitespace, replace with one
    return text


def remove_common_email_words(text):
    import re
    text = re.sub("\snotified by mailtrack le \w+ \w+ à \w+ \w+ a écrit", " ", text)
    text = re.sub('[Ee]nvoyé de mon \w+', " ", text)
    text = re.sub('bonjour\s', ' ', text)
    text = re.sub('\sallo|ô\s', ' ', text)
    text = re.sub('\senvoyé de mon \w+', ' ', text)
    text = re.sub('\sbonne journée\s', ' ', text)
    text = re.sub('\smerci\s', ' ', text)
    text = re.sub('\spour\s', ' ', text)
    text = re.sub('\setant\s', ' ', text)
    text = re.sub('\setait\s', ' ', text)
    text = re.sub('\splus\s', ' ', text)
    text = re.sub('\sbien\s', ' ', text)
    text = re.sub('https?', ' ', text)
    text = re.sub('www', ' ', text)  # 1992 count avec les donnees du 27-07-2020
    text = re.sub('\s(ca|com|org|fr)\s', ' ', text)
    text = re.sub('\swatchv?\s', ' ', text)  # apparait au dessus de 600 fois au total dans les emails
    text = re.sub('\s{2,}', ' ', text)  # if two or more whitespace, replace with one
    return text


def correct_word_for_dection(text):
    import re
    text = re.sub('\scovid19\s', 'covid 19', text)
    text = re.sub('\sfaceboux\s', 'facebook', text)
    text = re.sub('\sdr\s', 'docteur', text)
    return text


def clean_text_for_analysis(text):
    """
    Utiliser avant une analyse impliquant les mots pour retirés les éléments à ne pas compter.
    :param text:
    :return: text modifié
    """
    text = text.lower()
    text = remove_non_text_character(text)
    text = correct_word_for_dection(text)
    text = remove_common_email_words(text)
    text = remove_word_with_only_number(text)
    text = remove_single_letter(text)
    return text


def standardize_text(text):
    """correct encoding, remove emoji, non-roman character and extra space and tab"""
    text = encoding_correction(text)  # correction du style "=E9": "é"
    text = remove_emojis(text)
    text = remove_non_roman_character(text)  # grec, mandarin, arabe
    text = remove_extra_space(text)  # tab, double space,
    return text


def get_word_variation(word):
    return [word, find_french_feminine_of_word(word), find_french_plural_of_word(word)]

def words_in(words_to_find, text):
    """regarde si une combinaison de n mot est dans la liste de combinaison de mot. Fonctionne pour des ngrams de 2 ou 3
    :param words_to_find: string "je suis"
    :param text: text a decouper en ngram
    """
    import re
    in_list = False

    ngram_to_find = tuple(words_to_find.lower().split(' '))
    words = re.findall('\w+', text.lower())
    if len(ngram_to_find) == 2:
        list_ngram = zip(words, words[1:])
    elif len(ngram_to_find) == 3:
        list_ngram = zip(words, words[1:], words[2:])

    if ngram_to_find in list_ngram:
        in_list = True
    return in_list

def word_in(word_to_find, text):
    """
    Trouve le feminin et le pluriel du mot à trouver, converti le text en list et
    regarde si le mot, son feminin ou son pluriel est dans la liste
    :param word_to_find: Le mot à trouver dans text
    :param text: le texte dans lequel on regarde pour le mot
    :return: vrai si le mot, son féminin ou son pluriel se trouve dans text
    """
    in_list = False
    word_find_p = find_french_plural_of_word(word_to_find)
    word_to_find_f = find_french_feminine_of_word(word_to_find)
    word_to_find_f_p = find_french_plural_of_word(word_to_find_f)
    list_word_to_look_in = text.split()

    if word_to_find in list_word_to_look_in:
        in_list = True
    elif word_find_p in list_word_to_look_in:
        in_list = True
    elif word_to_find_f in list_word_to_look_in:
        in_list = True
    elif word_to_find_f_p in list_word_to_look_in:
        in_list = True
    return in_list

def string_in_text(s, text):
    """prend un string et retourne vrai si les mots sont dans texte. Appelle la fonction pour traiter plusieurs mots
     consecutif (3 max) si necessaire"""
    in_list = False
    if len(s.split()) > 1:
        in_list = words_in(s, text)
    else:
        #regarde pour un mot, mais aussi pour sa version feminin et pluriel
        in_list = word_in(s, text)
    return in_list

def find_french_plural_of_word(word):
    """retourne la version pluriel du mot selon les règles suivantes:
    https://grammaire.reverso.net/pluriel-des-noms-et-des-adjectifs/
    """
    word = word.lower()
    if len(word) >= 2 and word[-2:] == 'ou':
        if word in ['bijou', 'caillou', 'chou', 'genou', 'hibou', 'joujou', 'pou']:
            plural = word + 'x'

    elif len(word) >= 3 and word[-3:] == 'ail':
        if word in ["bail", "corail", "émail", "gemmail", "soupirail", "travail", "vantail", "vitrail"]:
            plural = word[:-3] + 'aux'

    elif word[-1:] in ['s', 'x', 'z']:
        plural = word

    elif (len(word) >= 3 and word[-3:] == 'eau') or (len(word) >= 2 and word[-2:] in ['au', 'eu']):
        plural = word + 'x'

    elif (len(word) >= 2 and word[-2:] == 'al'):
        plural = word[:-2] + 'aux'
    else:
        plural = word + 's'

    return plural


def find_french_feminine_of_word(word):
    """retourne la version du mot au féminin selon les règles suivantes :
    https://grammaire.reverso.net/la-formation-du-feminin/
    """
    word = word.lower()
    feminin = word
    if len(word) >= 2 and word[-2:] == 'er':
        feminin = word[:-2] + "ère"
    elif len(word) >= 1 and word[-1] == 'f':
        feminin = word[:-1] + 've'
    elif len(word) >= 3 and word[-3:] == 'eux':
        feminin = word[:-4] + 'euse'
    elif len(word) >= 2 and word[-2:] == 'el':
        feminin = word + 'le'
    elif len(word) >= 2 and word[-2:] == 'en':
        feminin = word + 'ne'
    elif len(word) >= 2 and word[-2:] == 'on':
        feminin = word + 'ne'
    elif len(word) >= 2 and word[-2:] == 'et':
        feminin = word + 'te'
    elif len(word) >= 3 and word[-3:] == 'eur':
        feminin = word[:-3] + 'euse'
    elif len(word) >= 4 and word[-4:] == 'teur':
        feminin = word[:-4] + 'trice'
    else:
        if word[-1:] not in ['e', 'x', 'z', 's']:
            feminin = word + "e"

    return feminin
