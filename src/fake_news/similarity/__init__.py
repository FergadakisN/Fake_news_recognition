"""Near-duplicate detection with MinHash and LSH, implemented from scratch (NumPy + standard library).

    shingles.py        text -> set of hashed 3-word shingles; exact Jaccard
    minhash.py         shingle set -> MinHash signature (estimates Jaccard)
    lsh.py             banding: signatures -> candidate pairs; LSHIndex (add / query)
    groups.py          verify candidates with exact Jaccard; union-find -> groups
    near_duplicates.py the pipeline on the cleaned data (cache, stages, summaries)
"""
