from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_it_chunk(name: str, chunk_size=200) -> str:
    f = open("../data/" + name, "r", encoding="utf-8")
    s = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    new_path = "../data/out_" + name
    docs = text_splitter.split_text(s)
    pf = open(new_path, "w", encoding="utf-8")
    for doc in docs:
        pf.write(doc + "\n" + "-"*30 + "\n")
    return new_path
