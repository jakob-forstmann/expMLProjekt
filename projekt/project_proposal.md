# Projekt Vorschlag

## zu lösendes Problem:
Stimmung eines Songs vorhersagen als Wert zwischen traurig(Wert 0 ) und fröhlich(Wert 1) vorhersagen.

## Datenset:
Das Datenset enthält Songs die zwischen 1960 und 2020 veröffentlicht wurden und auf Spotify zu finden sind.
Leider ist bei 10 Songs keine genaue release Datum angeben,aber das ist für das Projekt zweitrangig weil das release Datum nicht zur 
Vorhersage der Stimmung eines Songs genutzt wird.
Das Datenset ist verfügbar unter https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs

### Anzahl der Datenpunkte:
Das Datenset hat 23 verschiedene Attribute mit insgesamt 32.833 verschiedene Songs. 
Leider enthalten einige Songs für die valence Strings als Werte, sodass nach dem Aussortieren noch 32714 Songs übrig bleiben.
Von den 23 Attributen werden 5 für die Vorhersage genutzt werden.

### Attribute 

| track_id               | track_name                           | track_artist   | track_popularity | track_album_id         | track_album_name             | track_album_release_date                | playlist_name   | playlist_id            | playlist_genre | playlist_subgenre         | danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo  | duration_ms |
|------------------------|--------------------------------------|----------------|------------------|------------------------|------------------------------|-----------------------------------------|-----------------|------------------------|----------------|---------------------------|--------------|--------|-----|----------|------|-------------|--------------|------------------|----------|---------|--------|-------------|
| 29zWqhca3zt5NsckZqDf6c | Typhoon - Original Mix               | Julian Calor   | 27               | 0X3mUOm6MhxR7PzxG95rAo | Typhoon/Storm                | Tremor (Sensation 2014 Anthem)          | ♥ EDM LOVE 2020 | 6jI1gFr6ANFtT8MmTvA2Ux | edm            | progressive electro house | 603          | 884    | 5   | -4571    | 0    | 0.0385      | 1.33e-4      | 341              | 742      | 0.0894  | 127984 | 337500      |
| 7bxnKAamR3snQ1VGLuVfC1 | City Of Lights - Official Radio Edit | Lush & Simon   | 42               | 2azRoBBWEEEYhqV6sb7JrT | City Of Lights (Vocal Mix)   | The Best of Keith Sweat: Make You Sweat | ♥ EDM LOVE 2020 | 6jI1gFr6ANFtT8MmTvA2Ux | edm            | progressive electro house | 428          | 922    | 2   | -1814    | 1    | 0.0936      | 0.0766       | 0                | 0.0668   | 0.21    | 128.17 | 204375      |
| 5Aevni09Em4575077nkWHz | Closer - Sultan & Ned Shepard Remix  | Tegan and Sara | 20               | 6kD6KLxj7s8eCE3ABvAyf5 | Closer Remixed               | The Best of Keith Sweat: Make You Sweat | ♥ EDM LOVE 2020 | 6jI1gFr6ANFtT8MmTvA2Ux | edm            | progressive electro house | 522          | 786    | 0   | -4462    | 1    | 42          | 0.00171      | 0.00427          | 375      | 0.4     | 128041 | 353120      |
| 2m69mhnfQ1Oq6lGtXuYhgX | Only For You - Maor Levi Remix       | Mat Zo         | 15               | 1fGrOkHnHJcStl14zNx8Jy | Only For You (Remixes)       | The Best of Keith Sweat: Make You Sweat | ♥ EDM LOVE 2020 | 6jI1gFr6ANFtT8MmTvA2Ux | edm            | progressive electro house | 626          | 888    | 2   | -3361    | 1    | 109         | 0.00792      | 127              | 343      | 308     | 128008 | 367432      |
| 7ImMqPP3Q1yfUHvsdn7wEo | Sweet Surrender - Radio Edit         | Starkillers    | 14               | 0ltWNSY9JgxoIZO4VzuCa6 | Sweet Surrender (Radio Edit) | The Best of Keith Sweat: Make You Sweat | ♥ EDM LOVE 2020 | 6jI1gFr6ANFtT8MmTvA2Ux | edm            | progressive electro house | 529          | 821    | 6   | -4899    | 0    | 0.0481      | 108          | 1.11e-6          | 0.15     | 436     | 127989 | 210112      |


Bei den Beispieldaten fällt auf,dass die Attribute Tanzbarkeit und Stimmung laut README doubles zwischen 0 und 1 sein sollen tatsächlich aber vor allem  Werte zwischen 0 und 1000 auftauchen. Mein bisheriger Plan ist es, diese Werte in doubles zu verwandeln durch Division durch 1000. 

Von den 23 möglichen Attributen plane ich die die 5 Attribute :Geschwindigkeit in bpm,Tonart,Tanzbarkeit,mode(Dur oder Moll),Songname für die Vorhesage zu nuten.

## mögliche Algorithmen:
Da die Spalte "valence" jeden Wert zwischen 0 und 1 annehmen kann wie in dem Datenset sichtbar wurde, gibt es keine festen Klassen.
Es handelt sich also um ein Regression Problem.
mögliche Algorithen:
- Decision Tree
- Random Forest Classifier 
- Linear Regression bzw. SVMs  für Regression je nachdem wie viel Zeit noch bleibt.

## Evaluation des Modells:
Da es sich um ein Regression Problem handelt sind Metriken wie accuarcy sind leider nicht anwendbar da sie den Output des Modells mit der Gold Klasse vergleichen. Folgende Metriken sind stattdessen möglich:
- Mean Squared Error,Root Mean Squarred Error, quasi der gleich wie vorher nur mit einer Wurzel
- normalized Root Mean Squarred Error 
- Mean Absolute Error 
- R2 Score 

## erwartete Performanz:
Vermutlich lässt sich keine hohe Performanz erzielen, jedoch ist meine 
Hoffnung einen höheren Score als 0.5 zu erreichen. 
Denn ansonsten würden die  Vorhesagen 50 % von den tatsächlichen Werten 
abweichen. Dies entspricht dem Erwartungswert der Gleichverteilung zwischen 0 und 1,was bedeutet dass das Modell alle Werte als gleich wahrscheinlich betrachtet unabhängig von der Eingabe womit das Modell quasi einen Wert raten würde.
