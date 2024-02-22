# Exp Ml 1 Forstmann

## Beschreibung:
Das Repository enthält die Lösungen für die Übungsaufgaben 
und das Projekt für das Proseminar Experimente gestalten fürs maschinelle Lernen im WS 23/24. 
Jede Aufgabe befindet sich in einem extra Ordner mit einem eignenem README.
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

## Autor 
- Jakob Forstmann 
- forstmann@cl.uni-heidelberg.de
