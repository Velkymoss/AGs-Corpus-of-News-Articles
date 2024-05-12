import json
from utils import get_clean_data, get_train_dev_test_set

df, idx_label = get_clean_data()
train_set, dev_set, test_set = get_train_dev_test_set(df)

df.to_csv('data/clean_data.csv', index=False)
with open('data/idx_labels.json', 'w') as json_file:
    json.dump(idx_label, json_file, indent=4)
train_set.to_csv('data/train_set.csv', index=False)
dev_set.to_csv('data/dev_set.csv', index=False)
test_set.to_csv('data/test_set.csv', index=False)