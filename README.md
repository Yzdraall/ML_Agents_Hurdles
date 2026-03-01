# Deep Reinforcement Learning : 110m Hurdles

![Démo des agents en pleine course](ML_Agents_Hurdle_Demo.gif)

## Aperçu du Projet
Ce projet explore l'apprentissage par renforcement profond (Deep RL) appliqué à la physique continue. L'objectif : entraîner des agents articulés à maintenir leur équilibre, courir et sauter par-dessus des haies sur une piste de 110 mètres.

## Performances de l'IA
* **Entraînement massif :** Modèles stabilisés après **100 Millions d'étapes (steps)**.
* **Comportement émergent :** Les agents ont appris de manière autonome à coordonner leurs membres (Ragdoll physics) pour franchir les obstacles sans script d'animation prédéfini.
* **Inférence optimisée :** Modèles finaux intégrés via des réseaux de neurones `.onnx`.

## Les Agents
* **Walker Agent :** Agent bipède. Apprentissage complexe axé sur le balancier du corps et l'impulsion pour le saut.
* **Crawler Agent :** Agent quadrupède. Optimisation du centre de gravité et de la propulsion multi-membres.

## Stack Technique
* **Moteur 3D :** Unity (C#)
* **Intelligence Artificielle :** Unity ML-Agents Toolkit
* **Modélisation :** PyTorch / Exportation ONNX

## Comment tester le projet
1. Clonez ce dépôt.
2. Ouvrez le projet avec **Unity**.
3. Chargez la scène principale d'entraînement (ex: `Walker`).
4. Vérifiez que les modèles finaux `.onnx` (présents dans `results/v1.4_M/`) sont bien assignés dans le composant **Behavior Parameters** de l'agent.
5. Appuyez sur **Play** pour observer l'inférence en temps réel.
