# load the ecthr datasets from here, there are 3 choices:
# 1. do you want to look at allegations or violations?
# 2. do you want all facts or only the "silver" facts?
# 3. are you doing multi-label or binary classification?

from datasets import load_dataset
from datasets import Dataset, DatasetDict
import pandas as pd
from functools import partial

ARTICLES_DESC = {
    "2": "Right to life",
    "3": "Prohibition of torture",
    "4": "Prohibition of slavery and forced labour",
    "5": "Right to liberty and security",
    "6": "Right to a fair trial",
    "7": "No punishment without law",
    "8": "Right to respect for private and family life",
    "9": "Freedom of thought, conscience and religion",
    "10": "Freedom of expression",
    "11": "Freedom of assembly and association",
    "12": "Right to marry",
    "13": "Right to an effective remedy",
    "14": "Prohibition of discrimination",
    "15": "Derogation in time of emergency",
    "16": "Restrictions on political activity of aliens",
    "17": "Prohibition of abuse of rights",
    "18": "Limitation on use of restrictions on rights",
    "34": "Individual applications",
    "38": "Examination of the case",
    "39": "Friendly settlements",
    "46": "Binding force and execution of judgments",
    "P1-1": "Protection of property",
    "P1-2": "Right to education",
    "P1-3": "Right to free elections",
    "P3-1": "Right to free elections",
    "P4-1": "Prohibition of imprisonment for debt",
    "P4-2": "Freedom of movement",
    "P4-3": "Prohibition of expulsion of nationals",
    "P4-4": "Prohibition of collective expulsion of aliens",
    "P6-1": "Abolition of the death penalty",
    "P6-2": "Death penalty in time of war",
    "P6-3": "Prohibition of derogations",
    "P7-1": "Procedural safeguards relating to expulsion of aliens",
    "P7-2": "Right of appeal in criminal matters",
    "P7-3": "Compensation for wrongful conviction",
    "P7-4": "Right not to be tried or punished twice",
    "P7-5": "Equality between spouses",
    "P12-1": "General prohibition of discrimination",
    "P13-1": "Abolition of the death penalty",
    "P13-2": "Prohibition of derogations",
    "P13-3": "Prohibition of reservations",
}

ARTICLES_ID = {
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "11": 10,
    "12": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "34": 18,
    "38": 19,
    "39": 20,
    "46": 21,
    "P1-1": 22,
    "P1-2": 23,
    "P1-3": 24,
    "P3-1": 25,
    "P4-1": 26,
    "P4-2": 27,
    "P4-3": 28,
    "P4-4": 29,
    "P6-1": 30,
    "P6-2": 31,
    "P6-3": 32,
    "P7-1": 33,
    "P7-2": 34,
    "P7-3": 35,
    "P7-4": 36,
    "P7-5": 37,
    "P12-1": 38,
    "P13-1": 39,
    "P13-2": 40,
    "P13-3": 41,
}


def join_facts(example, silver=False):
    if not silver or len(example['silver_rationales']) == 0:
        facts = ' '.join(example['facts'])
        return {'facts': facts}
    facts = ' '.join([example['facts'][i] for i in example['silver_rationales']])
    return {'facts': facts}


def one_hot_encode(labels_list, num_classes=41):
    one_hot_vector = [0] * num_classes
    for label in labels_list:
        if label < num_classes:
            one_hot_vector[label] = 1
    return one_hot_vector


def multi_label(example):
    labels = example['labels']
    labels = [ARTICLES_ID[label] for label in labels]
    labels = one_hot_encode(labels)
    return {'labels': labels}


def binary_label(example):
    labels = example['labels']
    return {'labels': 1 if len(labels) > 0 else 0}


def load_ecthr_dataset(
        allegations: bool = True,
        silver: bool = False,
        is_multi_label: bool = True,
):
    if allegations:
        dataset = load_dataset("ecthr_cases", "alleged-violation-prediction", trust_remote_code=True)
    else:
        dataset = load_dataset("ecthr_cases", "violation-prediction", trust_remote_code=True)

    partial_join_facts = partial(join_facts, silver=silver)
    dataset = dataset.map(partial_join_facts)

    if is_multi_label:
        dataset = dataset.map(multi_label)
    else:
        if allegations:
            print("Binary classification for allegations doesnt seem sensible, but you do you")
        dataset = dataset.map(binary_label)

    dataset = dataset.remove_columns(['silver_rationales'])
    if allegations:
        dataset = dataset.remove_columns(['gold_rationales'])

    return dataset


def summarize_text_column(df, column_name, tokenizer=None):
    char_len = df[column_name].astype(str).str.len()
    print(f"Character Lengths of '{column_name}':")
    print(char_len.describe())

    if tokenizer:
        token_len = df[column_name].apply(lambda x: len(tokenizer.encode(x)))
        print(f"\nToken Lengths of '{column_name}' using the provided tokenizer:")
        print(token_len.describe())
    else:
        word_len = df[column_name].astype(str).str.split().str.len()
        print(f"\nWord Lengths of '{column_name}':")
        print(word_len.describe())