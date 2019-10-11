
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

c1 = 0
c2 = 0
c3 =0
c4 = 0
c5= 0
ratings = [1,2,3,4,5]
for d in dataset:
    if d['star_rating']  == 1:
        c1 += 1
    if d['star_rating']  == 2:
        c2 += 1
    if d['star_rating']  == 3:
        c3 += 1
    if d['star_rating']  == 4:
        c4 += 1
    if d['star_rating']  == 5:
        c5 += 1

counts = [c1,c2,c3,c4,c5]
counts

ratingsdist = dict(zip(ratings,counts))

X = list(ratingsdist.keys())
Y = [ratingsdist[x] for x in X]

print("Rating Distribution in dataset for " + str(ratings) +" is as follows:" + str(counts))
plt.xlabel("Count")
plt.ylabel("Rating")
plt.title("Rating Distribution in dataset")
plt.bar(X, Y)