# 04  Übung 

## Beschreibung
Dieser Ordner enthält die Projekt Beschreibung

## Beschreibung der Datensplits
Das Datenset ist nicht gesplittet in Train/Dev/Test Anteile. Um das Datenset zu splitten,müssen zuerst
alle nicht notwendigen Spalten gelöscht werden. Danach kann mit einer sklearn Methode das Datenset in 75% 
Trainingsdaten und 15% Testdaten aufgeteilt werden.
Da das Datenset wie oben beschrieben nicht in Train/Dev/Test Sets aufgeteilt ist aber das Dev Set für das 
Training der Hyperparameter benötigt wird werde ich K fold Cross Validierung benutzten. 
Obwohl dabei ein Teil des Trainingsdatensets für die Validierung genutzt werden muss ist dies immer noch besser
als ein neues Validierungsset suchen zu müssen, dass dann möglicherweise auch nicht die vorgebene Größe von 15% des 
ursprünglichen Datensets hat.

## Werteverteilung für die Spalte "valence"
![.Werte_verteilung](plots/valence_distribution.png)

In dem Plot geht das erste Intervall über 0 hinaus, da [pd.cut]( https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) aus der pandas Biblipthek zu dem ersten Intervall noch 0.1% dazuaddiert. 

| Wertebereich als linksoffenes Intervall      | Anzahl Datenpunkte
|----------------------------------------------|------|
| (0, 0.198]         | 3541 |
| (0.198, 0.396]     | 7668 |
| (0.396, 0.595]     | 9167 |
| (0.595, 0.793]     | 8012 |
| (0.793, 0.991]     | 4426 |
| gesamt             | 32814|


Es gibt insgesamt 1362 verschiedene Werte für die Valence, davon aber nur 68 verschiedene. Dabei treten 247 Werte nur einmal auf 
112 Werte nur 2x mal und 71 Werte nur 3x mal auf. 

Anzahl Datenpunkte pro Häufigkeit:
![Werteverteilung](plots/datapoints_per_frequency.png)

Die 20 häufigsten Werte für die Valence:
![Werteverteilung](plots/20_most_frequent_values.png)

## Evaluierung der Baselines:
Alle Metriken wurden mit 5 facher Kreuzvalidierung durchgeführt
Dabei benutzt sklearn für die Kreuzvalidierung den negatierten Fehler
für die Vorhersage auf dem Testdatenset aber den nicht negierten Fehler.

(negativer) root Mean squared error:
| Baseline |Durschnitt über 5 Folds| Score auf dem Testdatenset
|----------|:--------------|----|
| Mean Baseline| -0.2325| 0.2348|
| Majority Baseline|-0.5078|0.5056
| Random Baseline mit zufälliger valence 0.402| -0.2550 | 0.2598

(negativer) absolute error: 

| Baseline |Durschnitt über 5 Folds|Score auf dem Testdatenset
|----------|:--------------|----|
| Mean Baseline| -0.1954| 0.1976
| Majority Baseline|-0.4515| 0.4479
| Random Baseline zufälliger valence 0.402|-0.2119 |0.21558|


Beide Metriken messen wie groß die Differenz ist zwischen dem vorhergesagten valence und der tatsächlichen valence des aktuellen Songs angeben in rationalen Zahlen zwischen 0 und 1 ist. Eine niedriger Wert bedeutet dabei,dass die Vorhersage des Modells recht gut war da die Differenz zwischen dem vorhergesagten und der Gold valence gering ist.
Die Mean Baseline hat dabei für beide Metriken am besten performt, da der Durschnitt etwa 0.51 ist und wiederum im Intervall der häufigsten Werte ist.
Die Majority Baseline hat am schlechsten abgeschnitten denn nur der häufigst Wert für die Valence,0.961 der 69x mal vorkommt liegt nicht in dem liegen nicht in dem häufigsten Intervall.
Da außerdem 5 Werte die 68x mal vorkommen wieder genau in dem häufigsten Intervall in dem auch die Mean Baseline liegt vorkommen ist dies eine weiterer Grund für das schlechtere Abschneiden der Majority Baseline.


## Experimente  

### Feature Importance
Alle Feature Kombinationen wurden auf einem Decision Tree mit den beiden Evaluationsmetriken und 5-facher Kreuzvalidierung evaluiert.
Die angebenen Metriken sind  dabei der Durschnitt über die 5 Folds.

| Feature Kombination |RMSE| MEA| relativer Unterschied zu einem DT mit allen Features|
|---------------------| ---|-----|--------------------------------------------|
| danceability,track_album_name,tempo, loudness |-0,2142 | -0,1757 | < 0.01|
|track_album_name,tempo,loudness,mode           |-0,2257 |-0,1873  | < 0.05|
| danceability,track_album_name,tempo           |-0,2146 | -0,1762 | 
| danceability,tempo,mode                       |-0,2157 | -0,1771 |
| track_album_name,loudness,mode	            |-0,2306 | -0,1933 |



In der Tabelle sind jeweils die Kombination von 3 bzw. 4 verschiedenen Features mit den besten bzw. schlechtechsten RMSE bzw. MEA aufgeführt jeweils gerundet auf 4 Nachkommastellen. Die anderen Werte können in den Dateien im Ordner `evaluation_results` nachgelesen werden.

#### von sklearn berechnete Feauture Importance: 
| Feature | gerundete feature importance|
|---------|-----------------------------|
danceability |0.1092|
tempo|0.1596|
loudness|0.1168|
mode| 0.0092|


Die 10 wichtigsten Wörter für die Spalte track_album_name
| Wort | gerundeter TF-IDF Wert|
|------|-------------|
|the|0.0184|
|of|0.0079|
|deluxe|0.0061|
|feat|0.0060|
|hits| 0.0054|
|remix| 0.005|
|you| 0.005|
|edition| 0.0047|
|remastered| 0.0037|
|me|0.0036|

Wie erwartet ist es nicht verwunderlich, dass zwei Funktionswörter zu den 10 wichtigsten Wörtern zählen, da sie erstens allgemein sehr oft auftreten 
und deswegen auch entsprechend oft in Album Titeln vorkommt. 
Dies könnte vermieden werden, in dem man eine Liste an Stopwörtern benutzt.



### Optimierung der Hyperparameter für den Decision Tree: 
Der Paramter mean_samples_split hat keinen wesentlichen Einfluss auf die beiden Evalautionsmetriken:
| Metrik| gerundeter Durschnitt über 5 Folds | gerundete Standardabweichung bei 5 folds|
|-------|------------|--------------------|
| RMSE |-0.2142 | 0.0001; genauer Wert: 5.1346e-05|
| MEA | -0.1758 | 0.0001; genauer Wert: 5.1346e-05|

