import utils.pdf_processing
import pandas as pd

array = pd.DataFrame(utils.pdf_processing.make_it_readable("/home/vairisw/python/RAG-default/rag/data/digital_design_and_computer_architecture_russian_translation_July16.pdf"))
print(array.head())
