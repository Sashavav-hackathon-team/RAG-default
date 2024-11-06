from langchain_text_splitters import RecursiveCharacterTextSplitter

f = open("../data/input.txt", "r", encoding="utf-8")
s = f.read()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_text(s)
pf = open("Data/output.txt", "w", encoding="utf-8")
for doc in docs:
    pf.write(doc + "\n" + "-"*30 + "\n")
