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


def get_idx2lbl_lbl2idx(df: pd.DataFrame, column: str = "category") -> tuple[dict]:
    if column not in df.columns:
        raise ValueError(
            f"The dataframe does not contain the column '{column}'")
    category2lbl = {i: df[column].unique()[i]
                    for i in range(0, len(df[column].unique()))}
    lbl2category = {df[column].unique()[i]: i for i in range(
        0, len(df[column].unique()))}
    return category2lbl, lbl2category


def get_clean_data(filepath: str = "data/newsspace200.csv", min_token: int = 20, max_token: int = 250) -> pd.DataFrame:
    """
    Input:
        path to the newsspace200 csv file, file must contain the columns 'title', description' and 'category'
    Output:
        cleaned dataframe with new column description_token_length, title_token_length, article_token_length
    """
    if not filepath.endswith(".csv"):
        raise ValueError("The file must be a csv file")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"An error occurred trying to read the file: {str(e)}")
    if "title" not in df.columns:
        raise ValueError("The csv file does not contain the column 'title'")
    if "description" not in df.columns:
        raise ValueError(
            "The csv file does not contain the column 'description'")
    if "category" not in df.columns:
        raise ValueError("The csv file does not contain the column 'category'")
    df = df[["title", "description", "category"]]
    df = df[df['description'].notna() & df['title'].notna()]
    valid_categories = {category for category, frequency in df.category.value_counts(
    ).items() if frequency > 1000}
    # &(df.description.apply(lambda x: isinstance(x, str)))&(df.title.apply(lambda x: isinstance(x, str)))]
    df = df[df.category.isin(valid_categories)]
    df["description_token_length"] = [
        num_tokens_from_string(x) for x in df.description]
    df["title_token_length"] = [num_tokens_from_string(x) for x in df.title]
    df = df[df.title_token_length <= 100]
    df["article_token_length"] = df["description_token_length"] + \
        df["title_token_length"]
    # df["article"] = "title: " + df["title"] + " \n " + "description: " + df["description"]
    df["article"] = f"""
    title: {df["title"]}

    description: {df["description"]}
    """
    # filtering out articles with less than 20 tokens and more than 250 tokens
    df = df[(df.article_token_length >= min_token) &
            (df.article_token_length <= max_token)]
    return df


def get_train_dev_test_set(df: pd.DataFrame, threshold_minority_class: float = 0.01) -> tuple[pd.DataFrame]:
    if "category" not in df.columns:
        raise ValueError(
            "The dataframe does not contain the column 'category'")
    # getting frequency distribution of the category
    frequency_distribution = {category: (freq/(len(df)), freq)
                              for category, freq in df.category.value_counts().items()}
    # kicking out underrepresented classes
    underrepresented = {
        category for category in frequency_distribution if frequency_distribution[category][0] < threshold_minority_class}
    df = df[~df.category.isin(underrepresented)]

    # updating frequency distribution
    frequency_distribution = {category: (freq/(len(df)), freq)
                              for category, freq in df.category.value_counts().items()}
    # getting absolute frequency of the smallest class
    smallest_n = min(df.category.value_counts())
    # calcularing size of train, devtest and test set
    smallest_train_n = int(smallest_n * 0.7)
    devtest_n = int(len(df) * 0.15)
    # calculating absolute frequency distribution for devtest and test set
    dev_test_distribution = {
        category: int(devtest_n*freq[0]) for category, freq in frequency_distribution.items()}
    # creating datasets for train, dev and test per category
    category_datasets = {}
    for category, freq in dev_test_distribution.items():
        category_df = df[df.category == category]
        if category in ["World", "Entertainment", "Top Stories"]:
            num_samples = smallest_train_n + 28000 + 2*freq
            sample_indices = random.sample(
                range(len(category_df)), num_samples)
            train_indices = sample_indices[:smallest_train_n + 28000]
            dev_indices = sample_indices[smallest_train_n + 28000:(
                smallest_train_n + 28000 + (freq))]
            test_indices = sample_indices[(smallest_train_n + 28000 + freq):]

        else:
            num_samples = smallest_train_n + 2*freq
            sample_indices = random.sample(
                range(len(category_df)), num_samples)
            train_indices = sample_indices[:smallest_train_n]
            dev_indices = sample_indices[smallest_train_n:(
                smallest_train_n+(freq))]
            test_indices = sample_indices[(smallest_train_n+freq):]

        category_train = category_df.iloc[train_indices]
        category_dev = category_df.iloc[dev_indices]
        category_test = category_df.iloc[test_indices]
        category_datasets[category] = {"train": category_train,
                                       "dev": category_dev, "test": category_test}
    # concatenating the datasets
    train_set = pd.concat([category_datasets[category]["train"]
                          for category in category_datasets])
    dev_set = pd.concat([category_datasets[category]["dev"]
                        for category in category_datasets])
    test_set = pd.concat([category_datasets[category]["test"]
                         for category in category_datasets])

    return train_set, dev_set, test_set


def get_total_dataset(df: pd.DataFrame, threshold_minority_class: float = 0.01, min_token: int = 20, max_token: int = 250) -> pd.DataFrame:
    if "category" not in df.columns:
        raise ValueError(
            "The dataframe does not contain the column 'category'")
    # filtering out articles with less than 20 tokens and more than 250 tokens
    df = df[(df.article_token_length >= min_token) &
            (df.article_token_length <= max_token)]
    # getting frequency distribution of the category
    frequency_distribution = {category: (freq/(len(df)), freq)
                              for category, freq in df.category.value_counts().items()}
    # kicking out underrepresented classes
    underrepresented = {
        category for category in frequency_distribution if frequency_distribution[category][0] < threshold_minority_class}
    df = df[~df.category.isin(underrepresented)]
    return df
