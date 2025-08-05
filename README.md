# 🦯 DÉVELOPPEMENT D’UN SYSTÈME D’ASSISTANCE INTELLIGENT POUR LES PERSONNES AVEUGLES

## 📌 Description du projet
Ce projet consiste à développer un **système d’assistance intelligent** destiné aux personnes aveugles.  
Il combine plusieurs technologies de vision par ordinateur, de reconnaissance vocale et de détection d’obstacles pour offrir une aide en temps réel.  
L’objectif est de permettre une meilleure autonomie grâce à la détection d’objets, la reconnaissance faciale, la lecture de texte (OCR) et l’avertissement sonore.

---

## 🛠 Technologies et bibliothèques utilisées
- **Python**  
- **OpenCV** – Traitement d'images et vidéo  
- **YOLO** – Détection d’objets  
- **EasyOCR** – Reconnaissance optique de caractères (OCR)  
- **ESP32** – Communication avec les capteurs  
- **Capteur ultrason** – Détection d’obstacles  
- **Synthèse vocale** – Annonce des informations détectées

---

## 📂 Fichiers du projet

| Nom du fichier / dossier      | Description |
|------------------------------|-------------|
| `main.py`                    | Script principal qui intègre toutes les fonctionnalités |
| `object_detection.py`        | Détection d’objets avec YOLO |
| `face_recognition.py`        | Reconnaissance faciale |
| `ocr_reader.py`              | Lecture de texte avec EasyOCR |
| `color_detection.py`         | Détection des couleurs via HSV |
| `ultrasonic_sensor.py`       | Gestion du capteur ultrason |
| `esp32_comm.py`              | Communication avec l’ESP32 |
| `requirements.txt`           | Liste des dépendances Python |
| `result.png`                 | Exemple de résultat obtenu |

---

## ▶️ Comment exécuter le projet

1. **Installer les dépendances**
```bash
pip install -r requirements.txt
