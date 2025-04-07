import os
import re
import pandas as pd

def parse_text_files(base_path):
    """
    Parses text files from the specified folder structure into a pandas DataFrame.

    Args:
        base_path (str): Path to the dataset directory containing 'Cancer' and 'Non-Cancer' folders.

    Returns:
        pd.DataFrame: DataFrame with columns ['ID', 'Title', 'Abstract', 'Category']
    """
    data = []

    for category in ['Cancer', 'Non-Cancer']:
        folder_path = os.path.join(base_path, category)

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

                id_ = filename.replace('.txt', '')
                title = ''
                abstract = ''

                if len(lines) > 1:
                    title = lines[1].strip().replace('Title: ', '')
                    title = re.sub(r'^\W+|\W+$', '', title)
                if len(lines) > 2:
                    abstract = ' '.join(line.strip() for line in lines[2:])
                    abstract = abstract.replace('Abstract: ', '')
                    abstract = re.sub(r'^\W+|\W+$', '', abstract)

                data.append([id_, title, abstract, category])

    df = pd.DataFrame(data, columns=["ID", "Title", "Abstract", "Category"])
    df = df.dropna(subset=['Title', 'Abstract', 'Category'])
    return df