"""
Address will consist of:

level
flat
house
street
county
country
postcode

Example of full address:
    the 1st floor, 1C apartment, 35D-37D, Abbey North Road, Essex, England, 6XB IOR
    ----
    the = level_number_prefix
    1 = level_number
    st = level_number_suffix
    floor = level_type
    
    1 = flat_number
    C = flat_number_suffix
    apartment = flat_type
    
    35 -- number_first
    D -- number_first_suffix
    37 -- number_last
    D -- number_last_suffix
    
    Abbey = street
    North = street_suffix
    Road = street_type
    
    Essex = county
    England = state
    6XB IOR = postcode

Rules to vary addresses (how the dataset will be produced):

    1. LEVEL (optional, 10/90):
        level_number_prefix = the; optional - 20/80
        level_number = random number from 1 to 600; mandatory
        level_number_suffix = st, nd, rd, th; optional - 10/90
        level_type = random from level_type lookup; mandatory                                       (DIST 1)
        How to join?
            1. level_number_prefix, level_number, level_number_suffix, level_type
            2. level_type, level_number_prefix, level_number, level_number_suffix
    
    2. FLAT (mandatory):
        flat_number = random number from 1 to 600; optional (99/1)
        flat_number_suffix = random from A to J; optional (10/90)                                   (DIST 2)
        flat_type = random from flat_type lookup; optional (30/70)                                  (DIST 3)
        Constraints:
            IF FLAT NUMBER IS NOT PRESENT, FLAT NUMBER SUFFIX IS MANDATORY
        How to join?
            1. flat_number, flat_number_suffix, flat-type
            2. flat_type, flat_number, flat_number_suffix
    
    3. HOUSE NUMBER (optional, 15/85):
        number_first = random number from 1 to 600; mandatory
        number_first_suffix = random from A to Z; optional (10/90)                                  (DIST 2)
        number_last = random number to 600, > number_first; optional (20/80)
        number_last_suffix = random from A to Z; optional (10/90)                                   (DIST 2)
        How to join:
            1. number_first, number_first_suffix, number_last, number_last_suffix
    
    4. STREET (mandatory):
        street = random lookup from generated street list (scraper.py); mandatory
        street_suffix = random lookup from street_suffix_types, can be abbreviated; optional (10/90)
        street_type = random lookup from street_types, can be abbreviated; optional (70/30)
        How to join:
            any combination is possible (6 in total)
    
    5. GENERAL (mandatory):
        county = from postal_codes db; optional (25/75)
        state = from postal_codes db; optional (10/90)
        postcode = from postal_codes db; optional (70/30)
        How to join:
            any combination is possible (6 in total) 
    
    HOW TO JOIN (FINAL):
            LEVEL, FLAT, HOUSE NUMBER, STREET, GENERAL (most probable)
            FLAT, LEVEL, HOUSE NUMBER, STREET, GENERAL (rest)
            HOUSE NUMBER, STREET, LEVEL, FLAT, GENERAL (rest)
            GENERAL, STREET, HOUSE NUMBER, LEVEL, FLAT (rest)
    
    CONTRAINTS:
        Separator can be chosen either one for the whole address or separately for each different field
"""

from typing import Optional, Union, Callable, List
import random
from random import uniform
import string

import numpy as np

import lookup as lookups
from typo import generate_typo
from config.feature import *


# load config from globals()
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('src/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging



labels_list = [
    'level_number_prefix',  # 1 -- like "the", etc.
    'level_number',         # 2 -- like 1, 2, second, etc.
    'level_number_suffix',  # 3 -- like st, nd, rd, th, etc.
    'level_type',           # 4 -- like ground, floor, basement

    'flat_number',          # 5 -- number of the flat
    'flat_number_suffix',   # 6 -- letter after a flat (like A, B, C, etc.)
    'flat_type',            # 7 -- like "flat", "store", "apartment", etc.
    
    # number-first and number-last are numbers of houses where address is, something like: 35-37, 50-52
    'house_number_first',         # 8 -- first number
    'house_number_first_suffix',  # 9 -- sometimes a letter is included after (like 35c-36)
    'house_number_last',          # 10 -- last number
    'house_number_last_suffix',   # 11 -- letter after last number (80d-82d)

    'street_name',          # 12 -- name of the street
    'street_suffix_code',   # 13 -- like North, East, Upper, etc.
    'street_type_code',     # 14 -- street type (like road, street)

    'county',               # 15 -- county in UK

    'state',                # 16 -- state (like England, etc.)

    'postcode'              # 17 -- post code
]


####################################################################################
# Number of labels in total (+1 for the blank category)
n_labels = len(labels_list) + 1

# Allowable characters for the encoded representation
vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)


def vocab_lookup(characters: str):
    """
    Converts a string into a list of vocab indices
    :param characters: the string to convert
    :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return len(characters), np.array(result, dtype=np.int64)
####################################################################################


def random_separator(min_length: int = 1, max_length: int = 3, possible_sep_chars: Optional[str] = r",./\  ") -> str:
    """
    Generates a space-padded separator of random length using a random character from possible_sep_chars
    :param min_length: minimum length of the separator
    :param max_length: maximum length of the separator
    :param possible_sep_chars: string of possible characters to use for the separator
    :return: the separator string
    """
    chars = [" "] * random.randint(min_length, max_length)
    if len(chars) > 0 and possible_sep_chars:
        sep_char = random.choice(possible_sep_chars)
        chars[random.randrange(len(chars))] = sep_char
    return ''.join(chars)


def join_labels(lbls: [np.ndarray], sep: Union[str, Callable[..., str]] = " ") -> np.ndarray:
    """
    Concatenates a series of label matrices with a separator
    :param lbls: a list of numpy matrices
    :param sep: the separator string or function that returns the sep string
    :return: the concatenated labels
    """
    if len(lbls) < 2:
        return lbls

    joined_labels = None
    sep_str = None

    # if `sep` is not a function, set the separator (`sep_str`) to `sep`, otherwise leave as None
    if not callable(sep):
        sep_str = sep

    for l in lbls:
        if joined_labels is None:
            joined_labels = l
        else:
            # If `sep` is a function, call it on each iteration
            if callable(sep):
                sep_str = sep()

            # Skip zero-length labels
            if l.shape[0] == 0:
                continue
            elif sep_str is not None and len(sep_str) > 0 and joined_labels.shape[0] > 0:
                # Join using sep_str if it's present and non-zero in length
                joined_labels = np.concatenate([joined_labels, labels(sep_str, None, mutate=False)[1], l], axis=0)
            else:
                # Otherwise, directly concatenate the labels
                joined_labels = np.concatenate([joined_labels, l], axis=0)

    assert joined_labels is not None, "No labels were joined!"
    assert joined_labels.shape[1] == n_labels, "The number of labels generated was unexpected: got %i but wanted %i" % (
        joined_labels.shape[1], n_labels)

    return joined_labels


def join_str_and_labels(parts: [(str, np.ndarray)], sep: Union[str, Callable[..., str]] = " ") -> (str, np.ndarray):
    """
    Joins the strings and labels using the given separator
    :param parts: a list of string/label tuples
    :param sep: a string or function that returns the string to be used as a separator
    :return: the joined string and labels
    """
    # Keep only the parts with strings of length > 0
    parts = [p for p in parts if len(p[0]) > 0]

    # If there are no parts at all, return an empty string an array of shape (0, n_labels)
    if len(parts) == 0:
        return '', np.zeros((0, n_labels))
    # If there's only one part, just give it back as-is
    elif len(parts) == 1:
        return parts[0]

    # Pre-generate the separators - this is important if `sep` is a function returning non-deterministic results
    n_sep = len(parts) - 1
    if callable(sep):
        seps = [sep() for _ in range(n_sep)]
    else:
        seps = [sep] * n_sep
    seps += ['']

    # Join the strings using the list of separators
    strings = ''.join(sum([(s[0][0], s[1]) for s in zip(parts, seps)], ()))

    # Join the labels using an iterator function
    sep_iter = iter(seps)
    lbls = join_labels([s[1] for s in parts], sep=lambda: next(sep_iter))

    assert len(strings) == lbls.shape[0], "string length %i (%s), label length %i using sep %s" % (
        len(strings), strings, lbls.shape[0], seps)
    return strings, lbls


def labels(text: Union[str, int], field_name: Optional[str], mutate: bool = True) -> (str, np.ndarray):
    """
    Generates a numpy matrix labelling each character by field type. Strings have artificial typos introduced if
    mutate == True
    :param text: the text to label
    :param field_name: the name of the field to which the text belongs, or None if the label is blank
    :param mutate: introduce artificial typos
    :return: the original text and the numpy matrix of labels
    """

    # Ensure the input is a string, encoding None to an empty to string
    if text is None:
        text = ''
    else:
        # Introduce artificial typos if mutate == True
        text = generate_typo(str(text)) if mutate else str(text)
    labels_matrix = np.zeros((len(text), n_labels), dtype=np.bool_)

    # If no field is supplied, then encode the label using the blank category
    if field_name is None:
        labels_matrix[:, 0] = True
    else:
        labels_matrix[:, labels_list.index(field_name) + 1] = True
    return text, labels_matrix


def generate():
    for row in lookups.post_codes:
        level = generate_level()
        flat = generate_flat()
        house_number = generate_house_number()
        street = generate_street()
        general = generate_general()

        # join all together


def generate_level():
    level = {}
    
    # check if we skip the whole level
    if skip(config['level_prob']):
        return None

    # generate 'the'?
    if not skip(config['level_number_prefix_prob']):
        level['level_number_prefix'] = (labels('the', 'level_number_prefix'))
        
    # generate level number?
    if not skip(config['level_number_prob']):
        level['level_number'] = (labels(str(random.randint(1, 9)), 'level_number'))
        
        # generate postfix for level number?
        if not skip(config['level_number_suffix_prob']):
            level_suffix = lookups.num2suffix(level['level_number'][0])
            # append suffix to level number (but keep different label)
            level['level_number'] = join_str_and_labels([level['level_number'], (labels(level_suffix, 'level_number_suffix'))], sep='')
        
    
    if not skip(config['level_type_prob']):
        level['level_type'] = (labels(random.choice(lookups.level_types), 'level_type'))

    # decide how to mix it
    ordered_level = choose_join([
        
        # level_number_prefix, level_number, level_type
        list(filter(lambda x: x is not None, [
            level.get('level_number_prefix'), level.get('level_number'), level.get('level_type')
        ])),
        
        # level_type, level_number_prefix, level_number
        list(filter(lambda x: x is not None, [
            level.get('level_type'), level.get('level_number_prefix'), level.get('level_number')
        ]))
    ])
    return join_str_and_labels(ordered_level)


def generate_flat():
    """
    This functions generates a flat
    
    There are 4 configuration probabilities that can be tweaked:
    flat_prob - probability telling if flat will be generated at all
    flat_number_prob - probability that flat will have a number
    flat_number_suffix_prob - probability that flat will have a suffix ()
    flat_type_prob - probability that flat type will be generated (has to be looked up in lookups.flat_types)
    """
    flat = {}
    
    if skip(config.get('flat_prob', 0.99)):
        return None
    
    # generate flat number?
    if not skip(config.get('flat_number_prob', 0.99)):
        flat['flat_number'] = (labels(str(random.randint(1, 600)), 'flat_number'))
        
        # generate postfix for flat number?
        if not skip(config.get('flat_number_suffix_prob', 0.1)):
            flat_suffix = random.choice(string.ascii_uppercase[:10])
            flat['flat_number_suffix'] = (labels(flat_suffix, 'flat_number_suffix'))
    else:
        # If the flat number isn't present, then the suffix is mandatory
        flat_suffix = random.choice(string.ascii_uppercase)
        flat['flat_number_suffix'] = (labels(flat_suffix, 'flat_number_suffix'))

    # generate flat type?
    if not skip(config.get('flat_type_prob', 0.3)):
        flat['flat_type'] = (labels(random.choice(lookups.flat_types), 'flat_type'))
    
    # decide how to mix it
    ordered_flat = choose_join([
        # flat_number, flat_number_suffix, flat_type
        list(filter(lambda x: x is not None, [
            flat.get('flat_number'), flat.get('flat_number_suffix'), flat.get('flat_type')
        ])),
        
        # flat_type, flat_number, flat_number_suffix
        list(filter(lambda x: x is not None, [
            flat.get('flat_type'), flat.get('flat_number'), flat.get('flat_number_suffix')
        ]))
    ])
    
    return join_str_and_labels(ordered_flat)


def generate_house_number():
    """
    This function generates a house number.
    
    The configuration probabilities can be tweaked:
    house_number_prob - probability telling if a house number will be generated at all
    number_first_prob - probability that a house will have a first number
    number_first_suffix_prob - probability that first number will have a suffix
    number_last_prob - probability that a house will have a last number (greater than first number)
    number_last_suffix_prob - probability that last number will have a suffix
    """
    
    house_number = {}
    
    # check if we skip the whole house number
    if skip(config.get('house_number_prob', 0.85)):
        return None

    # generate number first
    if not skip(config.get('house_number_first_prob', 1.0)):  # Default is mandatory
        house_number['house_number_first'] = (labels(str(random.randint(1, 600)), 'house_number_first'))
        
        # generate postfix for first number
        if not skip(config.get('house_number_first_suffix_prob', 0.1)):
            number_first_suffix = random.choice(string.ascii_uppercase)
            house_number['house_number_first_suffix'] = (labels(number_first_suffix, 'house_number_first_suffix'))

    # generate number last
    if not skip(config.get('house_number_last_prob', 0.2)):
        # ensure the last number is greater than the first number
        first_num = int(house_number.get('house_number_first', [None, '0'])[0])
        house_number['house_number_last'] = (labels(str(random.randint(first_num + 1, 600)), 'house_number_last'))

        # generate postfix for last number
        if not skip(config.get('house_number_last_suffix_prob', 0.1)):
            number_last_suffix = random.choice(string.ascii_uppercase)
            house_number['house_number_last_suffix'] = (labels(number_last_suffix, 'house_number_last_suffix'))
    
    # decide how to mix it
    ordered_house_number_1 = filter(None, [
        house_number.get('house_number_first'), 
        house_number.get('house_number_first_suffix')
    ])
    
    ordered_house_number_2 = filter(None, [
        house_number.get('house_number_last'), 
        house_number.get('house_number_last_suffix')
    ])

    return join_str_and_labels(ordered_house_number_1, sep=''), join_str_and_labels(ordered_house_number_2, sep='')


def skip(prob):
    return prob <= uniform(0, 1)


def choose_join(combinations):
    return random.choice(combinations)


def generate_street():
    pass


def generate_general():
    pass


def skip(prob):
    return prob <= uniform(0, 1)


def choose_join(combinations):
    return random.choice(combinations)