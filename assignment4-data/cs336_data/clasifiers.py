import nltk
import fasttext


identify_language_model = fasttext.load_model("../data/classifiers/lid.176.bin")
nsfw_model = fasttext.load_model("../data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
hatespeech_model = fasttext.load_model("../data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")


def identify_language(text: str) -> tuple[any, float]:
    model = identify_language_model
    predictions = model.predict(text.replace("\n", " "), k=1)
    lang_label = predictions[0][0]
    confidence = predictions[1][0]
    language_code = lang_label.replace("__label__", "")
    return language_code, confidence


def clasify_nsfw(text: str) -> tuple[str, float]:
    model = nsfw_model
    prediction = model.predict(text.replace("\n", " "), k=1)
    label = prediction[0][0]
    confidence = prediction[1][0]
    label = label.replace("__label__", "")
    return label, confidence


def classify_hatespeech(text: str) -> tuple[str, float]:
    model = hatespeech_model
    prediction = model.predict(text.replace("\n", " "), k=1)
    label = prediction[0][0]
    confidence = prediction[1][0]
    label = label.replace("__label__", "")
    return label, confidence


def gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)

    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False
    
    total_length = sum(len(word) for word in words)
    avg_word_length = total_length / num_words
    if avg_word_length < 3 or avg_word_length > 10:
        return False
    
    lines = text.splitlines()
    ellipsis_count = sum(line.strip().endswith("...") for line in lines)
    if ellipsis_count / len(lines) > 0.3:
        return False
    
    alpha_count = sum(any(c.isalpha() for c in word) for word in words)
    if alpha_count / num_words < 0.8:
        return False
    
    return True