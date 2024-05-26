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
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "11": 9,
    "12": 10,
    "13": 11,
    "14": 12,
    "15": 13,
    "16": 14,
    "17": 15,
    "18": 16,
    "34": 17,
    "38": 18,
    "39": 19,
    "46": 20,
    "P1-1": 21,
    "P1-2": 22,
    "P1-3": 23,
    "P3-1": 24,
    "P4-1": 25,
    "P4-2": 26,
    "P4-3": 27,
    "P4-4": 28,
    "P6-1": 29,
    "P6-2": 30,
    "P6-3": 31,
    "P7-1": 32,
    "P7-2": 33,
    "P7-3": 34,
    "P7-4": 35,
    "P7-5": 36,
    "P12-1": 37,
    "P13-1": 38,
    "P13-2": 39,
    "P13-3": 40,
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


def frequency_threshold_labels(example, frequency_threshold, frequencies):
    labels = example['labels']
    return {'labels': [l for l in labels if frequencies[l] >= frequency_threshold]}


def load_ecthr_dataset(
        allegations: bool = True,
        silver: bool = False,
        is_multi_label: bool = True,
        frequency_threshold: int = 0
):
    frequencies = {}
    if allegations:
        dataset = load_dataset("ecthr_cases", "alleged-violation-prediction", trust_remote_code=True)
    else:
        dataset = load_dataset("ecthr_cases", "violation-prediction", trust_remote_code=True)

    partial_join_facts = partial(join_facts, silver=silver)
    dataset = dataset.map(partial_join_facts)

    for set in dataset:
        for example in dataset[set]:
            label = example['labels']
            if type(label) == int:
                label = [label]
            for l in label:
                if l not in frequencies:
                    frequencies[l] = 0
                frequencies[l] += 1

    partial_frequency_threshold = partial(frequency_threshold_labels, frequency_threshold=frequency_threshold, frequencies=frequencies)
    dataset = dataset.map(partial_frequency_threshold)

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

def tokenize_dataset(dataset, tokenizer, max_length=-1):
    def tokenize_function_truncate(examples, max_length=max_length):
        return tokenizer(examples['facts'], truncation=True, max_length=max_length)

    def tokenize_function(examples):
        return tokenizer(examples['facts'])

    if max_length >= 0:
        tokenized_dataset = dataset.map(tokenize_function_truncate, batched=True)
    else:
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

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
