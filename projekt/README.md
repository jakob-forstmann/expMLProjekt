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
als ein neues Validuerungsset suchen zu müssen, dass dann möglicherweise auch nicht die vorgebene Größe von 15% des 
ursprünglichen Datensets hat.

## Werteverteilung für die Spalte "valence"
![Werteverteilung](./werte_verteilung.png)

In dem Plot geht das erste Intervall über 0 hinaus, da [pd.cut]( https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) aus der pandas Biblipthek zu dem ersten Intervall noch 0.1% dazuaddiert. 

| Wertebereich als linksoffenes Intervall      | Anzahl Datenpunkte
|----------------------------------------------|------|
| (0, 0.198]         | 3541 |
| (0.198, 0.396]     | 7668 |
| (0.396, 0.595]     | 9167 |
| (0.595, 0.793]     | 8012 |
| (0.793, 0.991]     | 4426 |
| gesamt             | 32814|

## Evaluierung der Baselines:
Alle Metriken wurden mit 5 facher Kreuzvalidierung durchgeführt
Dabei negiert sklearn den Mean squared error und den absoluten Fehler
sodass höhere Fehler besser sind.

negativer Mean squared error auf dem Trainingsdatenset
| Baseline | Fehler| Durschnittlicher Fehler über alle 5 Folds|
|----------:|:-----|--------------|
| Mean Baseline| -0.23278676|-0.2325633174
| Majority Baseline|-0.23167599|-0.195415817
