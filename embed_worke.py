"""Egen prosess for spørsmåls-embeddinger (fastembed / ONNX).

Kjøres av app.py som underprosess. Hvis modellen krasjer eller går tom for
minne, dør bare denne prosessen; Streamlit-appen lever videre og faller
tilbake til nøkkelordsøk.

Protokoll: én JSON-linje inn per forespørsel {"texts": [...]},
én JSON-linje ut: {"vecs": [[...], ...]} eller {"error": "..."}.
Første linje ut er {"ready": true, "model": ...} når modellen er lastet.
"""
import json
import sys


def main():
    model_name = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=model_name, threads=threads)
        # Én prøvekjøring slik at ONNX-sesjonen er helt initialisert.
        list(model.embed(["oppstart"]))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Kunne ikke laste modell: {type(exc).__name__}: {exc}"}), flush=True)
        return
    print(json.dumps({"ready": True, "model": model_name}), flush=True)
    for line in sys.stdin:
        try:
            texts = json.loads(line)["texts"]
            vecs = [v.tolist() for v in model.embed(texts)]
            print(json.dumps({"vecs": vecs}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
