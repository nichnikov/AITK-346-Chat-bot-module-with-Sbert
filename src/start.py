import os
import fasttext
from src.texts_processing import TextsTokenizer
from src.config import (stopwords,
                        parameters,
                        logger,
                        PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier

print("PROJECT_ROOT_DIR", PROJECT_ROOT_DIR)

ft_model = fasttext.load_model(os.path.join(PROJECT_ROOT_DIR, "models", "bss_cbow_lem.bin"))
tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, ft_model)
logger.info("service started...")
