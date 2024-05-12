import pandas as pd
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_clean_data(filepath: str = "data/newsspace200.csv") -> pd.DataFrame:
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
    return df