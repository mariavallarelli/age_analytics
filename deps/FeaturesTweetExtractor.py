import re
from datetime import datetime
import emoji


def assign_range(age):
    """ This method assign a numeric value to the target variable."""
    if age:
        if int(age) <= 29.0:
            return 1
        elif int(age) <= 39.0 and age >= 30.0:
            return 2
        elif int(age) <= 49.0 and age >= 40.0:
            return 3
        elif int(age) <= 59.0 and age >= 50.0:
            return 4
        elif int(age) >= 60.0:
            return 5


def assign_range_binary(age):
    """ This method assign a binary value to the target variable."""
    return 0 if age <= 29.0 else 1


def find_happy_birthday_to_someone_else(text):
    """ This method find all sentences which contains i.e.'happy 21st birthday to someone else'."""
    matches = re.findall(r"@*.*Happy\D*\d{2}[th|rd|st].*birthday.*@*\w+", text, re.IGNORECASE)
    return ",".join(matches) if matches else ""


def count_url(text):
    """ This method counts url present in the text of tweet """
    matches = re.findall(r'https?://\S+', text)
    return len(matches)


def count_exclamation_question_period_marks(text):
    """ This method counts exclamation/question and period marks present in the text of tweet """
    pattern = r"[.?!]"
    sentences = [i for i in re.split(pattern, text) if text]
    if len(sentences) > 0:
        return len(sentences)
    else:
        return 0


def count_mentions(text):
    """ This method counts mentions to twitter users present in the text of tweet"""
    matches = re.findall(r"@(\w+)", text)
    return len(matches)


def find_year_of_birth(text):
    """ This method find some birth years parsing twitter screennames."""
    matches = re.findall(r"19[5-9][0-9]$", text, re.IGNORECASE)
    if len(matches) == 0:
        matches = re.findall(r"\D+\d{2}$", text, re.IGNORECASE)  # maria72
        if len(matches) == 0:  # si assume che non vi siano utenti che usano twitter con età superiore a 70 anni
            return ""
        else:
            matches = re.findall(r"\d{2}$", matches[0], re.IGNORECASE)
            if int(matches[0]) >= 50:
                return matches[0]
            else:
                return ""
    else:
        return matches[0]


def count_hash_tags(text):
    """This  method counts hashtags present in the text of tweet"""

    matches = re.findall(r"#(\w+)", text)
    return len(matches) if matches else 0


def find_happy_birthday_tome(text):
    """ This method find all sentences which contains i.e.'happy 21st birthday to me'."""
    matches = re.findall(r"Happy\D*\d{2}[th|rd|st].*birthday.*to me", text, re.IGNORECASE)
    return ",".join(matches) if matches else ""


def find_birthday_screen_name(row):
    """ This method find all twitter screen_names whose tweet text contains i.e. happy 21 st birthday
    to me / someone else or with screen_names with birth year"""
    text_1 = row['_owner_birthday_wishes']
    # text_2 = row['_birthday_wishes']
    text_3 = row['_year_of_birth']
    matches = []
    if text_1 or text_3:
        screen_name = row['user-screen_name']
        # print(screen_name)
        return screen_name
    '''
    if text_2:
        # print(text_2)
        matches = re.findall(r"@\w+", text_2)
        # print(matches)
        return str(matches[0].replace("@", "")) if matches else ""
    '''


def try_parsing_date(date_string):
    # print(datestring)
    return date_string.strftime('%d/%m/%Y') if date_string else ""


def get_age(row):
    """ This method adjust the age comparing the tweet creation date with sysdate"""
    # Happy Birthday to me
    text_1 = row['_owner_birthday_wishes']
    # Happy birthday to someone else
    # text_2 = row['_birthday_wishes']

    # Year from user-screen_name
    text_3 = row['_year_of_birth']
    dt = row['tweet_created_at']
    created_at_year = datetime.strptime(dt, '%a %b %d %H:%M:%S %z %Y').year
    matches = []
    if text_1:
        matches = re.findall(r"\d{2}", text_1)
        year_diff = int(datetime.today().year) - int(created_at_year)
        return int(matches[0]) + year_diff
    elif text_3:
        if len(text_3) == 2:
            return int(datetime.today().year) - (1900 + int(text_3))
        elif len(text_3) == 4:
            return int(datetime.today().year) - (int(text_3))
    else:
        return ""
    '''
    elif text_2:
        matches = re.findall(r"\d{2}", text_2)
        year_diff = int(datetime.today().year) - int(created_at_year)
        return int(matches[0]) + year_diff
    '''


def count_emoticon(str):
    """This  method counts emoticon present in text of tweet """
    return len([c for c in emoji.UNICODE_EMOJI if c in str])


def count_personal_pronouns(text):
    """This  method counts personal pronouns present in text of tweet """
    # pattern = r"[I|you|he|she|it|we|they|me|him|her|us|them]"
    pattern = (r'\we\b | \bthey\b | \bI\b | \byou\b | \bthey\b | \bhe\b | \bshe\b | \bit\b | \bme\b | \bhim\b | \bher\b | \bus\b | \bthem\b | \bits\b')
    res = re.findall(pattern, text, re.IGNORECASE)
    if len(res) > 0:
        return len(res)
    else:
        return 0