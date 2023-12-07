# 04  Übung 

## Beschreibung
Dieser Ordner enthält die Projekt Beschreibung für Übung 04.

# Beschreibung der Datensplits
Das Datenset ist nicht gesplittet in Train/Dev/Test Anteile. Um das Datenset zu splitten,müssen zuerst
alle nicht notwendigen Spalten gelöscht werden. Danach kann mit einer sklearn Methode das Datenset in 75% 
Trainingsdaten und 15% Testdaten aufgeteilt werden.
Da das Datenset wie oben beschrieben nicht in Train/Dev/Test Sets aufgeteilt ist aber das Dev Set für das 
Training der Hyperparameter benötigt wird werde ich K fold Cross Validierung benutzten. 
Obwohl dabei ein Teil des Trainingsdatensets für die Validierung genutzt werden muss ist dies immer noch besser
als ein neues Validuerungsset suchen zu müssen, dass dann möglicherweise auch nicht die vorgebene Größe von 15% des 
ursprünglichen Datensets hat.


# Werteverteilung für die Spalte "valence"
Jeder Datenpunkt kann ein Label zwischen 0 und 1 jeweils inklusive annehmen.
Aus diesem Grund ist mir unklar geblieben wie ich diese Werte plotten soll.
Eine Gruppierung nach der Häufigkeit aufgeteilt auf die 4 Kriterien hat ergeben,dass
bei etwa 0.6 die meisten Kombinationen von Trainigspunkten liegen.Dann gibt es aber noch etliche
Kombinationen an Trainingspunkten die nur einmal vorkommen






