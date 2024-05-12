import random
import pandas as pd
import tiktoken

seed_value = 42
random.seed(seed_value)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_clean_data(filepath: str = "data/newsspace200.csv") -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Input:
        path to the newsspace200 csv file, file must contain the columns 'title', description' and 'category'
    Output:
        cleaned dataframe with new column description_token_length, title_token_length, article_token_length
    """
    if not filepath.endswith(".csv"):
        raise ValueError("The file must be a csv file")
    try:
        df =  pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"An error occurred trying to read the file: {str(e)}")
    if "title" not in df.columns:
        raise ValueError("The csv file does not contain the column 'title'")
    if "description" not in df.columns:
        raise ValueError("The csv file does not contain the column 'description'")
    if "category" not in df.columns:
        raise ValueError("The csv file does not contain the column 'category'")
    df = df[["title", "description", "category"]]
    df = df[df['description'].notna() & df['title'].notna()]
    valid_categories = {category for category, frequency in df.category.value_counts().items() if frequency > 1000}
    df = df[df.category.isin(valid_categories)]#&(df.description.apply(lambda x: isinstance(x, str)))&(df.title.apply(lambda x: isinstance(x, str)))]
    df["description_token_length"] = [num_tokens_from_string(x) for x in df.description]
    df["title_token_length"] = [num_tokens_from_string(x) for x in df.title]
    df = df[df.title_token_length <= 100]
    df["article_token_length"] = df["description_token_length"] + df["title_token_length"]
    df["article"] = "title: " + df["title"] + "description: " + df["description"]
    df["labels"], label_strings = pd.factorize(df['category'])
    idx_str_category = {i:l for (i,l) in zip([l for l in range(len(label_strings))], label_strings)}
    return df, idx_str_category

def get_train_dev_test_set(df: pd.DataFrame, threshold_minority_class: float = 0.01) -> tuple[pd.DataFrame]:
    if "labels" not in df.columns:
        raise ValueError("The dataframe does not contain the column 'labels'")
    # getting frequency distribution of the labels
    frequency_distribution = {idx: (freq/(len(df)), freq) for idx, freq in df.labels.value_counts().items()}
    # kicking out underrepresented classes
    underrepresented = {idx for idx in frequency_distribution if frequency_distribution[idx][0] < threshold_minority_class}
    df = df[~df.labels.isin(underrepresented)]
    # updating frequency distribution
    frequency_distribution = {idx: (freq/(len(df)), freq) for idx, freq in df.labels.value_counts().items()}
    # getting absolute frequency of the smallest class
    smallest_n = min(df.labels.value_counts())
    # calcularing size of train, devtest and test set
    smallest_train_n = int(smallest_n * 0.7)
    train_n = int(smallest_train_n * len(frequency_distribution))
    devtest_n = int(train_n * 0.15)
    # calculating absolute frequency distribution for devtest and test set
    dev_test_distribution = {idx: int(devtest_n*freq[0]) for idx, freq in frequency_distribution.items()}
    # creating datasets for train, dev and test per category
    category_datasets = {}
    for idx, freq in dev_test_distribution.items():
        category_df = df[df.labels == idx]
        num_samples = smallest_train_n + 2*freq
        sample_indices = random.sample(range(len(category_df)), num_samples)
        train_indices = sample_indices[:smallest_train_n]
        dev_indices = sample_indices[smallest_train_n:(smallest_train_n+(freq))]
        test_indices = sample_indices[(smallest_train_n+freq):]
        category_train = category_df.iloc[train_indices]
        category_dev = category_df.iloc[dev_indices]
        category_test = category_df.iloc[test_indices]
        category_datasets[idx] = {"train": category_train, "dev": category_dev, "test": category_test}
    # concatenating the datasets
    train_set = pd.concat([category_datasets[idx]["train"] for idx in category_datasets])
    dev_set = pd.concat([category_datasets[idx]["dev"] for idx in category_datasets])
    test_set = pd.concat([category_datasets[idx]["test"] for idx in category_datasets])
    return train_set, dev_set, test_set
        