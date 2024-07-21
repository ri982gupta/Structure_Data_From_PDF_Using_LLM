# from transformers import BertTokenizer, BertForTokenClassification
# import torch
# import pandas as pd

# def extract_entities(text):
#     # Load pre-trained BERT model and tokenizer
#     model_name = 'bert-base-uncased'
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForTokenClassification.from_pretrained(model_name)

#     # Tokenize input text
#     tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
#     inputs = tokenizer(text, return_tensors='pt')

#     # Predict labels for each token
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=2)

#     # Decode predicted labels into entity spans
#     entities = []
#     entity_type = None
#     entity_start = None

#     for i in range(len(predictions[0])):
#         token = tokens[i]
#         label = predictions[0][i].item()

#         if label != 0:  # 0 is the label for 'O' (outside of any entity)
#             label_name = model.config.id2label[label]
#             if label_name != 'O':
#                 if entity_type is None:
#                     entity_type = label_name
#                     entity_start = i
#                 elif label_name == entity_type:
#                     continue
#                 else:
#                     entity_end = i
#                     entities.append((entity_type, entity_start, entity_end))
#                     entity_type = label_name
#                     entity_start = i

#     # Format entities into a structured format (e.g., DataFrame)
#     structured_data = pd.DataFrame(columns=['Entity Type', 'Text'])
#     for entity in entities:
#         entity_type, start, end = entity
#         text = ' '.join(tokens[start:end+1])
#         structured_data = structured_data.append({'Entity Type': entity_type, 'Text': text}, ignore_index=True)

#     return structured_data

# def main():
#     input_file = 'input.txt'
#     with open(input_file, 'r') as file:
#         raw_text = file.read()

#     structured_data = extract_entities(raw_text)
#     print("Structured Data:")
#     print(structured_data)

# if __name__ == "__main__":
#     main()


# GETTING ERROR


# C:\Users\RiGupta\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<?, ?B/s]
# C:\Users\RiGupta\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py:157: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\RiGupta\.cache\huggingface\hub\models--bert-base-uncased. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
#   warnings.warn(message)
# vocab.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 557kB/s]
# tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 771kB/s]
# config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 570/570 [00:00<?, ?B/s]
# model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 440M/440M [00:41<00:00, 10.6MB/s]
# Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# Token indices sequence length is longer than the specified maximum sequence length for this model (3877 > 512). Running this sequence through the model will result in indexing errors
# 2024-07-11 19:05:31.024246: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-07-11 19:06:40.326187: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# object address  : 000001F006C20AC0
# object refcount : 3
# object type     : 00007FFD7CAAD910
# object type name: KeyboardInterrupt
# object repr     : KeyboardInterrupt()

#-----------------------------------------HANDLE TOKEN SIZE-----------------------------------------------------------


# from transformers import BertTokenizer, BertForTokenClassification
# import torch
# import pandas as pd

# def extract_entities(text):
#     # Load pre-trained BERT model and tokenizer
#     model_name = 'bert-base-uncased'
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForTokenClassification.from_pretrained(model_name)

#     # Split text into smaller segments to avoid sequence length issue
#     max_seq_length = 512
#     segments = [text[i:i+max_seq_length] for i in range(0, len(text), max_seq_length)]

#     entities = []

#     # Process each segment with BERT
#     for segment in segments:
#         # Tokenize input text
#         inputs = tokenizer(segment, return_tensors='pt', truncation=True, padding=True)

#         # Predict labels for each token
#         with torch.no_grad():
#             outputs = model(**inputs)

#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=2)

#         # Decode predicted labels into entity spans
#         for i in range(len(predictions)):
#             tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[i])
#             labels = predictions[i]

#             entity_start = None
#             entity_type = None

#             for j, label_id in enumerate(labels):
#                 label_name = model.config.id2label[label_id.item()]
#                 if label_name.startswith('B-'):
#                     if entity_type is not None:
#                         entities.append((entity_type, entity_start, j-1))
#                     entity_type = label_name[2:]
#                     entity_start = j
#                 elif label_name.startswith('I-'):
#                     continue
#                 else:
#                     if entity_type is not None:
#                         entities.append((entity_type, entity_start, j-1))
#                         entity_type = None
#                         entity_start = None

#     # Format entities into a structured format (e.g., DataFrame)
#     structured_data = pd.DataFrame(columns=['Entity Type', 'Text'])
#     for entity in entities:
#         entity_type, start, end = entity
#         text = text[start:end+1]
#         structured_data = structured_data.append({'Entity Type': entity_type, 'Text': text}, ignore_index=True)

#     return structured_data

# def main():
#     input_file = 'input.txt'
#     with open(input_file, 'r') as file:
#         raw_text = file.read()

#     structured_data = extract_entities(raw_text)
#     print("Structured Data:")
#     print(structured_data)

# if __name__ == "__main__":
#     main()

# GETTING ERROR

# C:\Users\RiGupta\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# Structured Data:
# Empty DataFrame
# Columns: [Entity Type, Text]
# Index: []


#------------------------------------------------------------------------------------------------------------------------------


from transformers import BertTokenizer, BertForTokenClassification
import torch
import pandas as pd

def extract_entities(text):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    # Split text into smaller segments to avoid sequence length issue
    max_seq_length = 512
    segments = [text[i:i+max_seq_length] for i in range(0, len(text), max_seq_length)]

    entities = []

    # Process each segment with BERT
    for segment in segments:
        # Tokenize input text
        inputs = tokenizer(segment, return_tensors='pt', truncation=True, padding=True)

        # Predict labels for each token
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        # Decode predicted labels into entity spans
        for i in range(len(predictions)):
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[i])
            labels = predictions[i]

            entity_start = None
            entity_type = None

            for j, label_id in enumerate(labels):
                label_name = model.config.id2label[label_id.item()]
                if label_name.startswith('B-'):
                    if entity_type is not None:
                        entities.append((entity_type, tokenizer.convert_tokens_to_string(tokens[entity_start:j])))
                    entity_type = label_name[2:]
                    entity_start = j
                elif label_name.startswith('I-'):
                    continue
                else:
                    if entity_type is not None:
                        entities.append((entity_type, tokenizer.convert_tokens_to_string(tokens[entity_start:j])))
                        entity_type = None
                        entity_start = None

    # Format entities into a structured format (e.g., DataFrame)
    structured_data = pd.DataFrame(entities, columns=['Entity Type', 'Text'])
    
    return structured_data

def main():
    input_file = 'input.txt'
    with open(input_file, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    structured_data = extract_entities(raw_text)
    print("Structured Data:")
    print(structured_data)

if __name__ == "__main__":
    main()
