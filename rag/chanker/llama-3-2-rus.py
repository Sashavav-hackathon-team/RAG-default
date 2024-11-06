import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загрузка модели и токенизатора
model_name = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка входного текста
input_text = ("У меня 20 друзей. "
              "По приведенной выше информации ответь на вопрос: "
              "Сколько у меня друзей?")

# Токенизация и генерация текста
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output = model.generate(
  input_ids,
  max_length=1512,
  temperature=0.3,
  num_return_sequences=1,
  no_repeat_ngram_size=2,
  top_k=50,
  top_p=0.95,
  )

# Декодирование и вывод результата
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
