import fasttext


class QualityModel:
    def __init__(
        self, 
        input_path: str="../data/quality/2k_training_data.txt",
        model_path: str="../data/classifiers/wiki_cc_classifier.bin"
    ) -> None:
        self.input_path = input_path
        self.model_path = model_path

    def train(self):
        model = fasttext.train_supervised(
            input=self.input_path,
            lr=0.2,
            epoch=30,
            wordNgrams=2,
            verbose=2,
        )
        model.save_model(self.model_path)
        self.model = model
    
    def load_model(self):
        self.model = fasttext.load_model(self.model_path)

    def inference(self, text: str) -> tuple[str, float]:
        prediction = self.model.predict(text.replace("\n", " "), k=1)
        label = prediction[0][0]
        confidence = prediction[1][0]
        label = "wiki" if label == "__label__positive" else "cc"
        return label, confidence
