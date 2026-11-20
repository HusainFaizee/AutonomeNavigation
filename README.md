Reinforcement Learning: GridWorld

Deutsch

Ein Python-basiertes Reinforcement-Learning-Projekt zur Implementierung und zum Vergleich von drei Reinforcement-Learning-Algorithmen in einer selbst entwickelten GridWorld-Umgebung.

Implementierte Algorithmen

Q-Learning: Ein wertbasierter, Off-Policy-Reinforcement-Learning-Algorithmus.

SARSA: Ein wertbasierter, On-Policy-Reinforcement-Learning-Algorithmus.

Deep Q-Network (DQN): Ein Deep-Reinforcement-Learning-Algorithmus, der ein neuronales Netzwerk zur Approximation von Aktionswerten verwendet.

Projektstruktur

agents/ – Implementierungen der Q-Learning-, SARSA- und DQN-Agenten.

env/ – Selbst entwickelte GridWorld-Umgebung.

training/ – Trainingsskripte für die einzelnen Algorithmen.

utils/ – Hilfsfunktionen und Visualisierungen.

results/ – Grafische Darstellungen der Trainingsergebnisse.

config.py – Konfiguration der Experimente.

main.py – Hauptprogramm zur Ausführung der Experimente.

experiment_runner.py – Ausführung der Experimente.

Verwendete Technologien

Python

NumPy

PyTorch

Matplotlib

Installation

Repository klonen:

git clone YOUR_REPOSITORY_URL
cd MLGridProject

Abhängigkeiten installieren:

pip install -r requirements.txt

Ausführung

python main.py

Ergebnisse

Der Ordner results/ enthält Visualisierungen der Experimente, darunter Ergebnisse für DQN und Q-Learning.

Mögliche zukünftige Erweiterungen

Vergleich mit weiteren Reinforcement-Learning-Algorithmen.

Erweiterte Hyperparameter-Optimierung.

Umfangreichere experimentelle Auswertungen.

English

A Python-based reinforcement learning project implementing and comparing three reinforcement learning algorithms in a custom-built GridWorld environment.

Implemented Algorithms

Q-Learning: A value-based, off-policy reinforcement learning algorithm.

SARSA: A value-based, on-policy reinforcement learning algorithm.

Deep Q-Network (DQN): A deep reinforcement learning algorithm that uses a neural network to approximate action values.

Project Structure

agents/ – Implementations of the Q-Learning, SARSA and DQN agents.

env/ – Custom-built GridWorld environment.

training/ – Training scripts for the individual algorithms.

utils/ – Helper functions and visualisation utilities.

results/ – Graphical representations of training results.

config.py – Experiment configuration.

main.py – Main program for running experiments.

experiment_runner.py – Experiment execution.

Technologies Used

Python

NumPy

PyTorch

Matplotlib

Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL
cd MLGridProject

Install dependencies:

pip install -r requirements.txt

Running the Project

python main.py

Results

The results/ directory contains visualisations of the experiments, including DQN and Q-Learning results.

Potential Future Improvements

Comparison with additional reinforcement learning algorithms.

Extended hyperparameter optimisation.

More comprehensive experimental evaluation.