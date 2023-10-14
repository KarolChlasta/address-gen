import requests
import re


letter_links = [f'https://www.streetlist.co.uk/gazetteer/{chr(n)}.html' for n in range(97, 123)]
out = './data/streets.csv'

name_regex = re.compile(r'<a href="https:\/\/www\.streetlist\.co\.uk\/road-name\/.*?"*?>(.*?)<\/a>')
roads = []

for link in letter_links:
    print(f'Processing {link}')
    resp = requests.get(link)
    raw = name_regex.findall(resp.text)
    
    for name in raw:
        if '(' in name:
            processed_name = name.split('(')[0].strip()
        else:
            processed_name = ' '.join(name.split(' ')[:-1])
        
        road_names = processed_name.split('/')
        for name in road_names:
            if name.strip() != '':
                roads.append(name)


roads = list(sorted(set(roads)))[1:]

with open(out, 'w+') as f:
    f.write(f'ind\tstreet\n')
    for i, road in enumerate(roads):
        f.write(f'{i}\t{road}\n')
