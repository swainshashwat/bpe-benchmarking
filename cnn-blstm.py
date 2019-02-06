from bpemb import BPEmb
from tqdm import tqdm
import csv
import numpy
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
bpemb_en=BPEmb(lang="en",dim=50)

#print(bpemb_en.encode("Stratford"))
#print(bpemb_en.embed("Stratford").shape)

with open("datasets/Chatbot/train.csv") as f:
    reader = csv.reader(f, delimiter='\t')
    max_len = 0
    y = []
    for row in tqdm(reader):
        y.append(row[1])
        sample_len = len(bpemb_en.encode(row[0]))
        max_len = sample_len if sample_len > max_len else max_len

#print(max_len)
#print(y[:10])

# label encoder
le = LabelEncoder()
encoded_labels = le.fit_transform(y)
#print(encoded_labels)
print(le.classes_)

x = None

y = to_categorical(encoded_labels, num_classes=len(le.classes_))
#print(y)

with open("datasets/Chatbot/train.csv") as f:
    reader = csv.reader(f, delimiter='\t')
    for row in tqdm(reader):
        embeddings = bpemb_en.embed(row[0])
        