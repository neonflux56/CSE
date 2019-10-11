
import gzip
import csv
import matplotlib.pyplot as plt

c = csv.reader(gzip.open("amazon_reviews_us_Gift_Card_v1_00.tsv.gz", 'r'), delimiter = '\t')
dataset = []
first = True
for line in c:
    # The first line is the header
    if first:
        header = line
        first = False
    else:
        d = dict(zip(header, line))
        # Convert strings to integers for some fields:
        d['star_rating'] = int(d['star_rating'])
        d['helpful_votes'] = int(d['helpful_votes'])
        d['total_votes'] = int(d['total_votes'])
        dataset.append(d)


len(dataset)