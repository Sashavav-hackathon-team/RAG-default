from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("Data/MaiaSandu.txt") as f:
    state_of_the_union = f.read()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_text(state_of_the_union)
for doc in docs:
    print(doc, end="\n-----------------------------------\n")
