from utils import get_clean_data, get_train_dev_test_set, get_total_dataset

df = get_clean_data()
df.to_csv('data/clean_data_2.csv', index=False)
#train_set, dev_set, test_set = get_train_dev_test_set(df)

# train_set.to_csv('data/train_set.csv', index=False)
# dev_set.to_csv('data/dev_set.csv', index=False)
# test_set.to_csv('data/test_set.csv', index=False)

# totat_dataset = get_total_dataset(df)
# totat_dataset.to_csv('data/total_dataset.csv', index=False)
