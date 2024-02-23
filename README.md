# Exp Ml 1 Forstmann

## Beschreibung:
Das Repository enthält die Lösungen für die Übungsaufgaben 
und das Projekt für das Proseminar Experimente gestalten fürs maschinelle Lernen im WS 23/24. 
Jede Aufgabe befindet sich in einem extra Ordner mit einem eigenem README.
Der Ordner Projekt enthält das Datenset,den Code und die Ergebnisse sowie Plots für das Projekt.

## Übersicht:
- [Übung 1](01_übung)
- [Übung 2](02_übung)
- [Übung 3](03_übung)
- [Projekt](projekt)


## Einrichtung:
``` 
git clone https://gitlab.cl.uni-heidelberg.de/forstmann/exp-ml-1-forstmann.git

# optional: virtual Environment erstellen
python -m venv .venv  
source .venv/bin/activate
``` 
Um das Projekt ausführen zu können, kann es notwendig sein die Umgebungsvariable `PYTHONPATH`
nachträglich auf die entsprechenden Pfade zu setzten. Dies kann  unter Linux und in der WSL 
z.b. so erreicht werden:
``` 
PYTHONPATH="/path/to/projekt":$PYTHONPATH 
PYTHONPATH="/path/to/projekt/models":$PYTHONPATH 
PYTHONPATH="/path/to/projekt/experiments":$PYTHONPATH 
PYTHONPATH="/path/to/projekt/feature_ablation":$PYTHONPATH 
PYTHONPATH="/path/to/projekt/spotify_API":$PYTHONPATH 
export PYTHONPATH
``` 

Um mithilfe der Spotify API Songs Empfehlungen zu generieren sind folgende Schritte notwendig:

0. ggf.Spotify Account erstellen(egal ob premium oder free Account)
1. Spotify CLIENT ID und CLIENT SECRET generieren, eine Anleitung gibt es [hier](https://developer.spotify.com/documentation/web-api)
2. eine neue Datei `API_Keys.py` im Ornder `spotify_API` erstellen und die CLIENT ID und das CLIENT SECRET dort hin kopieren


## Vorraussetzungen 
Um den Code für das Projekt und die Übungen ausführen zu können 
müssen git und python 3.11 oder eine ähnliche python Version installiert sein. Die benötigten Bibliothen für die Übungen bzw. für das Projekt stehen jeweils in 2 unterschiedlichen `requirements.txt` Dateien.
Die Vorrausetzungen können mit den folgenden Schritten installiert werden.
```
# installiere die notwendigen Bibliotheken für die Übungen
pip install -r excercise_requirements.txt

# Oder installiere die notwendigen Bibliotheken für das Projekt 
# und die Übungen 
pip install -r projekt/project_requirements.txt
```

## Benutzung:
Der Code für die einzelnen Übungen kann direkt ausgeführt werden nachdem die Voraussetzungen installiert sind.
Für das Projekt gibt es mehrere Möglichkeiten:

1. Für die Ergebnisse der verwendeten Modelle kann die `main.py` in dem Order `projekt` wie folgt aufgerufen werden:
    ````
    # model name ersetzen oder weglassen,dann wird als Default Modell die Majority Baseline genutzt 
    python main.py --model {model_name}
    # verfügbare Modelle anzeigen
    python  main.py --help
    ````
2. Die Dateien im Ordner `projekt` individuell ausführen,dies ist bei fast allen Dateien möglich

## Autor 
- Jakob Forstmann 
- forstmann@cl.uni-heidelberg.de
