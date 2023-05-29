import os
# import fasttext
from src.texts_processing import TextsTokenizer
# from texts_processing import TextsTokenizer
from src.config import (stopwords,
# from config import (stopwords,
                        parameters,
                        logger,
                        PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier
# from classifiers import FastAnswerClassifier
from sentence_transformers import SentenceTransformer

print("PROJECT_ROOT_DIR", PROJECT_ROOT_DIR)

model = SentenceTransformer(os.path.join("models", "expbot_paraphrase.transformers"))
tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, model)
logger.info("service started...")
