"""
Fast Indic Transliteration Caching for India Entities.
Uses text_unidecode to transliterate native Indic scripts (Hindi, Tamil, Telugu, Gujarati, Bengali, etc.)
into normalized Latin ASCII tokens for high-speed indexing and string matching.
"""

import sys
import time
import polars as pl
import text_unidecode

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def transliterate_indic_series(series: pl.Series) -> pl.Series:
    """
    Transliterates a Polars Series of text strings from Indic to Latin ASCII.
    Only processes non-ASCII strings to maximize performance.
    """
    values = series.fill_null("").to_list()
    out = []
    for s in values:
        if any(ord(c) > 127 for c in s):
            out.append(text_unidecode.unidecode(s))
        else:
            out.append(s)
    return pl.Series(name=series.name, values=out)


if __name__ == "__main__":
    t0 = time.time()
    print("Testing transliteration on 10,000 Indian sample records...")
    s1_in = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t").filter(
        pl.col("country") == "India"
    ).head(10000)
    
    trans_names = transliterate_indic_series(s1_in["business_name"])
    print(f"Transliterated 10,000 records in {time.time()-t0:.2f}s.")
    
    # Inspect 5 transliterated samples
    for i in range(min(5, len(s1_in))):
        orig = s1_in["business_name"][i]
        trans = trans_names[i]
        if orig != trans:
            print(f"Orig: {orig}  -->  Trans: {trans}")
