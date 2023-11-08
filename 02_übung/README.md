# 02_Übung

# Beschreibung
Dieser Ordner enthält die Lösungen für das Übungsblatt 2 It`s time for Machine Learning



# Aufgabe 1: 
- Zeile 30 - 38: initalisiere die benutzten  ML Methoden mit ihren jeweiligen Parametern
    - Knearest Neighbor bei für die Klassifikation eines Punktes jeweils seine 3 Nachbarn berücksichtigt werden 
    - SVC mit einem linaren Kernel und einem regularization parameter $C=0.025$
    - SVC mit einem rbf Kernel mit $\gamma$=2,regularization parameter $C=1$
    - Decision Tree Classifier 
    - Random Forest Classifier 
    - Ein neuronales Netz das mit 1000 Iterationen über die Trainingdaten trainiert wird 
    - Gaussian Naive Bayes


- Zeile 39: lasse ein zufälliges Datenset von sklearn erzeugen mit zwei nicht redudanten
  Feautures und zwei Klassen die jeweils ein Cluster haben

- Zeile 45: erzeuge drei weitere Datensets mit sklearn und füge das in Zeile 39 erzeugte Datenset hinzu, jedes Datenset besteht aus einem Array von samples und dem dazugehörige Klassenlabel 

- Zeile 56: hier findet das Preprocessing statt: diese Daten werden als Vektoren repräsentiert und anschließend wird der Mittelwert der Repräsentationen abgezogen und schlussendlich skaliert  mit der Standardvarianz der gewählten Repräsentation 

- Zeile 57: teilt das Datenset auf: $\frac{3}{4}$ des Datenset werden Trainingsdaten und der Rest sind Testdaten. Beide sind aufgeteilt in die tatsächlichen Daten und ein dem Daten zugewiesenes Label.

- Zeile 60-62: erzeugt ein (x_max -x_min) * (y_max-y_min) großes Rechteck das jede Kombinastion von x und y zwischen  (x_max -x_min) und (y_max -y_min) enthält.
Dieses Rechteck wird später genutzt um darauf die Trainings- und Testdaten zu zeichnen.

- Zeile 66-81: Zeichnet die Trainings- und Testdaten auf dem vorher erstellten Rechteck und nimmt noch eine Achseneinteilung und eine Achsenbeschriftung vor

- Zeile 84 -115: iteriert über jeden oben erstellen Classifier und den Namen und
  plottet die Decision boundary,die Trainings und Testdaten und  nimmt noch eine Achseneinteilung und eine Achsenbeschriftung vor. 
  Zwei Zeilen sind in dem Abschnitt dabei besonders relevant:
  - Zeile 86: trainiere den in der aktuellen Iteration ausgewählten Classifier auf den Trainingsdaten
  - Zeile 87: hier wird evaluiert, dafür wird die durschnittliche Accuary(=Genauigkeit) des in der aktuellen Iteration ausgewählten Classifier auf den Testdaten berechnet