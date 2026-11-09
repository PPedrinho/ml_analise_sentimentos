
import joblib
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"


def main():
    model = joblib.load(MODEL_PATH)

    textos = [
        "Adorei o produto, excelente qualidade!",
        "Eu não amei.",
        "Péssimo atendimento, não recomendo.",
        "O produto chegou ontem pela manhã."
    ]

    previsao = model.predict(textos)

    for Comentario, sentimento in zip(textos, previsao):
        logger.info(f"Comentario: {Comentario}")
        logger.success(f"Previsão do sentimento: {sentimento}")
        print("-" * 50)


if __name__ == "__main__":
    main()
