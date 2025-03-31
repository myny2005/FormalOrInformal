# FormalOrInformal

## requirements:
- transformers>=4.36.0
- datasets>=2.16.0 
- torch>=2.0.0
- scikit-learn>=1.3.0
- pandas>=1.5.0
- tqdm>=4.66.0
- wikipedia-api>=0.6.0
- praw>=7.7.0

## data  
In the `data` folder there are 2 notebooks which scrape informal data from Reddit and formal data from Wikipedia.  
If you want to scrape your own dataset, you'll need to use your own `praw` and Wikipedia API agent (user-agent string) 
But don’t worry — to test the model, the provided `.txt` files are totally enough.  

There’s also a CSV file (`labeled_texts.csv`) that contains both types of texts, already labeled:
- `0` → formal  
- `1` → informal  

## scripts

In the `scripts/` folder you'll find:

- `data_preparation.py`  
  Loads raw `.txt` files, labels the data, and outputs a single CSV you can use for training.

- `simple.ipynb`  
  Trains a simple **TF-IDF + Logistic Regression** model. Surprisingly effective for such a small and fast solution.

- `fine_tune.ipynb`  
  Fine-tunes `distilbert-base-uncased` using Hugging Face Transformers.  
  Trained on **CPU only**, so the dataset is small — but the code is scalable. You can always add more data.

---

This article is worth mentioning and it helped me a lot!
https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c

## results
This model does incredibly well and hardly ever makes any mistakes!
You’ll find saved models in the `models/` directory:
- `distilbert/` → the fine-tuned DistilBERT model  

You can run inference with:

```python
from transformers import pipeline

model = pipeline("text-classification", model="./models/distilbert")

print(model("This method demonstrates high academic rigor."))
# likely: formal

print(model("yo this is wild af lmao"))
# likely: informal
```
The models file is more than 100MB so I provide a link to it using Google Drive:
https://drive.google.com/drive/u/0/folders/1DhLhtR3vyYP3RdBtCBmFtM-wImK-RYAU
