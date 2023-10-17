# One Warehouse multi retailers with production constraints
Travaux effectués durant mon stage de recherche (M1) d'avril à juillet 2023.

Mon stage de recherche portait sur l'étude du problème One Warehouse Multi Retailer (OWMR). C'est un problème de lot-sizing à deux niveaux. Mes travaux avaient pour but de formuler des nouvelles contraintes pour les commandes à l'entrepôt et de comparer les différentes formulations obtenues.

Le code est en Python et fait appel au solveur CPLEX.

Importe le repo : 
```ruby
git clone https://github.com/agathelhm/Agathe_LHERMITE_entretien_technique_localsolver.git
```

Packages utilisés : 
```ruby
pip install -r requirements.txt
```

Contient quatre dossiers :
- data contient les instances utilisées (10 instances sont déjà stockées)
- Generates_instances permet de génerer de nouvelles instances, à la fois sous un format .dat et à partir de ce format, sous un format .npy
- Run_multiprocessing contient les différentes formulations du problèmes ainsi que les nouvelles contraintes implémentées. Les donnes obtenues sont ensuite enregistrées dans des fichiers csv
- Heuristic contient une implémentation naive d'une heuristique pour résoudre le probleme OWMR lorsque l'entrepot a des contraintes sur ses périodes de production.

Les scripts pythons appellant les modèles et enregistrant les données sont :
- pour Run_multiprocessing, multiprocess_test.py
- pour Heuristic, multiprocess_random_periods.py
 Ces fichiers enregistrent les informations de la resolution des problèmes dans les sous-dossiers results.


  Le fichier Book_of_abstracts.pdf contient l'abstract étendu de 4 pages que j'avais fait pour présenter mes résultats à l'International Workshop on Lot-Sizing (IWLS 2023) (page 65).
  Le fichier rapport contient mon rapport de stage complet (environ 40 pages).
  
