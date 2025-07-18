#!/usr/bin/env python3
import argparse
import boto3
import pandas as pd
import io

def download_df(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def upload_df(s3, df, bucket, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def preprocess(df):
    # 1) Rename the incoming columns exactly as in your file
    if 'Class Index' in df.columns:
        df = df.rename(columns={
            'Class Index': 'label',
            'Title':       'title',
            'Description': 'description'
        })
    # 2) Shift labels 1–4 → 0–3
    df['label'] = df['label'] - 1
    # 3) Concatenate title + description, drop any nulls
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df = df.dropna(subset=['text','label'])
    # 4) Keep only the two columns we need downstream
    return df[['label','text']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket',     required=True, help='S3 bucket name')
    parser.add_argument('--in-prefix',  required=True, help='raw data prefix')
    parser.add_argument('--out-prefix', required=True, help='clean data prefix')
    args = parser.parse_args()

    s3 = boto3.client('s3')
    bkt = args.bucket
    inp = args.in_prefix.rstrip('/')
    out = args.out_prefix.rstrip('/')

    # Build the S3 keys based on your folder structure + filenames
    train_key = f"{inp}/train.csv"
    test_key  = f"{inp}/test.csv"

    # Download
    train_raw = download_df(s3, bkt, train_key)
    test_raw  = download_df(s3, bkt, test_key)

    # Preprocess
    train_clean = preprocess(train_raw)
    test_clean  = preprocess(test_raw)

    # Upload cleaned files back under your specified folder
    upload_df(s3, train_clean, bkt, f"{out}/train_clean.csv")
    upload_df(s3, test_clean,  bkt, f"{out}/test_clean.csv")

    print("✅ Done. Cleaned files at:")
    print(f"   s3://{bkt}/{out}/train_clean.csv")
    print(f"   s3://{bkt}/{out}/test_clean.csv")

if __name__=='__main__':
    main()
