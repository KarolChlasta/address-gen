# from config.feature import *

# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging


# from dataset import labels, join_str_and_labels
# str_name, str_name_label = labels('Abbey', 'street_name')
# str_type, str_type_label = labels('Road', 'street_type_code')

# print(type(labels('Abbey', 'street_name')))
# # print(str_type, str_type_label)

# print(join_str_and_labels([
#     (str_name, str_name_label),
#     (str_type, str_type_label)
# ])[1].shape)

from dataset import generate_level, generate_flat, generate_house_number

print(generate_house_number())