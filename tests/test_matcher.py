from screener.matcher import rank_resumes_against_jd

def test_basic():
    jd = "We are hiring a Python developer with experience in machine learning and SQL."
    resumes = [
        "Alice has 3 years of Python and ML experience. She knows TensorFlow and SQL.",
        "Bob is a graphic designer with Adobe skills.",
    ]
    # simulate paths by writing temp files
    import tempfile, os
    rpaths = []
    for i, txt in enumerate(resumes):
        p = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        p.write(txt.encode("utf-8")); p.flush()
        rpaths.append(p.name)
    df = rank_resumes_against_jd(jd, rpaths, use_spacy=False)
    assert df.iloc[0].similarity >= df.iloc[1].similarity
