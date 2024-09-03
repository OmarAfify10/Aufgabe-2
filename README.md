# Aufgabe-2
Angegleichungsleistung
# Project 
In diesem Projekt wurde ein Datensatz aus dem Internet heruntergeladen, um ein maschinelles Lernmodell (ML) zu trainieren. Wir haben die Leistung des Modells anhand der Genauigkeit und einer Konfusionsmatrix gemessen. Die Ergebnisse wurden in train_metrics.pkl und training_time.pkl gespeichert. Wir haben meinen Logger und meinen Timer verwendet, um die Genauigkeit und die Laufzeit des Modells jederzeit zu überprüfen. Das Projekt zeigt, ob die aktuellen Ergebnisse mit den vorherigen übereinstimmen oder nicht.
# Ausführen 
1)Öffnen Sie ein Python-Terminal in Binder: Starten Sie, indem Sie ein neues Python-Terminal in Binder öffnen.

2)Führen Sie das Trainingsskript aus: Führen Sie das erste Skript aus, um das maschinelle Lernmodell zu trainieren, indem Sie den folgenden Befehl eingeben:'python train_model.py' Dadurch wird eine Datei (train_metrics.pkl) erstellt, die die Genauigkeit und die Konfusionsmatrix speichert.

3)Führen Sie das Testskript aus: Nach dem Training führen Sie das Testskript aus, um die aktuellen Metriken mit den zuvor gespeicherten Werten zu vergleichen:'python test_model.py' Dieses Skript vergleicht die aktuelle Genauigkeit und die Konfusionsmatrix mit den Werten, die in train_metrics.pkl gespeichert sind.

4)Messen Sie die Trainingszeit: Um die Trainingszeit zu messen, führen Sie den folgenden Befehl aus: 'python train_time.py
'Dies berechnet die Trainingszeit und speichert sie in einer Datei namens training_time.pkl.

5)Testen Sie die Trainingszeit: Vergleichen Sie schließlich die neue Trainingszeit mit der zuvor gespeicherten Zeit:   'python test_time.py'
'python test_time.py'
Dieser Befehl zeigt das neue Ergebnis der Laufzeit an und gibt an, ob es mit den zuvor aufgezeichneten Werten übereinstimmt.


# Binder Badge 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OmarAfify10/Aufgabe-2/HEAD)
