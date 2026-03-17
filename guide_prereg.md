# Guide de Pré-enregistrement
> Ce qu'on attend d'un pré-reg — du protocole à l'analyse  
> Recherche empirique en sciences cognitives, NLP & psychiatrie

---

## 0. Pourquoi pré-enregistrer ?

Le pré-enregistrement consiste à soumettre — **avant la collecte ou l'analyse des données** — un document décrivant les hypothèses, le design et le plan d'analyse. L'objectif est de distinguer de manière transparente ce qui était prévu a priori (analyses **confirmatives**) de ce qui a été découvert après coup (analyses **exploratoires**).

**Avantages :**
- Réduire le HARKing (*Hypothesizing After Results are Known*)
- Limiter le p-hacking et la flexibilité analytique non déclarée
- Augmenter la crédibilité des résultats nuls
- Faciliter la revue par les pairs et la reproductibilité
- Parfois obligatoire pour les Registered Reports ou certains financeurs

> ⚠️ Un pré-reg n'empêche pas les analyses exploratoires. Il exige simplement qu'elles soient **identifiées comme telles** dans le papier.

---

## 1. Vue d'ensemble — Checklist

| Section | Obligatoire | Recommandé | Plateforme(s) |
|---|:---:|:---:|---|
| Titre & auteurs | ✅ | | Toutes |
| Hypothèses | ✅ | | Toutes |
| Variables (VI/VD) | ✅ | | Toutes |
| Design & paradigme | ✅ | | Toutes |
| Participants & critères | ✅ | | Toutes |
| Calcul de puissance | | ✅ | AsPredicted, OSF |
| Plan d'analyse statistique | ✅ | | Toutes |
| Analyses exploratoires | | ✅ | OSF, AsPredicted |
| Données manquantes | | ✅ | OSF |
| Corrections multiples | | ✅ | Toutes |
| Code d'analyse (draft) | | ✅ | OSF |

---

## 2. Sections détaillées

### 2.1 Titre, auteurs & contexte

| Champ | Contenu attendu |
|---|---|
| Titre de l'étude | Court, informatif. Peut inclure le paradigme ou la population. |
| Auteurs | Noms et affiliations. Certaines plateformes permettent l'anonymisation. |
| Date de soumission | Timestamp automatique sur OSF/AsPredicted — ne pas modifier a posteriori. |
| Statut des données | Données déjà collectées ? En cours ? Pas encore commencée ? |
| Lien pré-reg antérieur | Si amendement d'un pré-reg existant, indiquer le lien. |

> 💡 Sur AsPredicted, le contexte est intégré dans la Q1 : « Avez-vous déjà collecté des données ? ». Répondez honnêtement — c'est la base de la transparence.

---

### 2.2 Hypothèses

C'est la section la plus importante du pré-reg. Les hypothèses doivent être formulées **avant de regarder les résultats** et de manière suffisamment précise pour être **falsifiables**.

#### Caractéristiques d'une bonne hypothèse

- **Directionnelle si possible** : précisez le sens attendu (e.g., « le groupe SSD présentera un ASPL plus élevé que les contrôles »)
- **Opérationnalisée** : liée à une mesure concrète et au test statistique utilisé
- **Distinguée** : séparer clairement les hypothèses primaires des secondaires/exploratoires
- **Numérotée** : facilite la référence croisée dans le papier

#### Structure recommandée

```
TH1 (confirmative, primaire) :
Le groupe SSD présente un degré moyen (Average Degree) significativement inférieur
au groupe contrôle, évalué par un test de Welch bilatéral (α = .05).

TH2 (confirmative, secondaire) :
L'ASPL est corrélé négativement avec le score PANSS positif dans le groupe SSD
(corrélation de Spearman, unilatérale, α = .05).

TH3 (exploratoire) :
Des différences de LSCC pourraient émerger en fonction du sous-type diagnostique,
sans direction prédéfinie.
```

> ⚠️ Ne reformulez pas vos hypothèses après avoir vu les données. Si vous devez les ajuster, soumettez un **amendement daté** sur OSF.

---

### 2.3 Variables

Toutes les variables utilisées dans les analyses confirmatives doivent être définies **avant** que les données ne soient analysées.

#### Variables indépendantes (VI)
- Nature : catégorielle (groupe : SSD vs contrôle) ou continue (score clinique)
- Opérationnalisation : comment est-elle mesurée ou assignée ?
- Niveaux : si catégorielle, lister tous les niveaux

#### Variables dépendantes (VD)
- Nom de la métrique (e.g., LSCC, Average Degree, ASPL)
- Mode de calcul : logiciel, version, paramètres (e.g., taille de fenêtre pour le graphe de parole)
- Transformation prévue : log, z-score, normalisation ?
- Unité de mesure et range attendu

#### Covariables et variables de nuisance
- Variables contrôlées dans les modèles (âge, sexe, niveau d'éducation, durée de maladie…)
- Justifier leur inclusion **a priori**, pas après coup

> 💡 Pour les graphes de parole : précisez la taille de la fenêtre glissante, le seuil de connexion, et la version du script utilisé.

---

### 2.4 Design expérimental & paradigme

| Champ | Contenu attendu |
|---|---|
| Type d'étude | Observationnelle, expérimentale, quasi-expérimentale, longitudinale… |
| Paradigme | e.g., STST, entretien libre, tâche de fluence… |
| Conditions / groupes | Liste des groupes, randomisation éventuelle |
| Variables between/within | Facteurs inter-sujets vs intra-sujets |
| Protocole d'enregistrement | Durée de la production, mode (audio/vidéo), environnement |
| Aveugle (blinding) | Les annotateurs / analyseurs sont-ils en aveugle du groupe ? |

---

### 2.5 Participants

#### Critères d'inclusion / exclusion
- Critères diagnostiques utilisés (DSM-5, CIM-11, PANSS…)
- Âge, langue maternelle, niveau d'éducation
- Exclusions : comorbidités, traitements, données manquantes au-delà d'un seuil

#### Taille d'échantillon & calcul de puissance

Le calcul de puissance est **obligatoire** pour les études confirmatives. Il doit être fait a priori et documenté.

- **Effet attendu** (d de Cohen, η², r…) : justifié par la littérature ou une étude pilote
- **Puissance cible** : généralement ≥ .80 (souvent .90 pour les études cliniques)
- **Seuil α** : habituellement .05, ou corrigé si tests multiples
- **Logiciel utilisé** : G*Power, R (`pwr`), Python (`pingouin`)…
- **N total** et par groupe

> 💡 Si le dataset est déjà collecté (e.g., N = 119), précisez-le explicitement et justifiez via une **analyse de sensibilité** (*power sensitivity analysis*) plutôt qu'un calcul a priori classique.

---

### 2.6 Plan d'analyse statistique

C'est la section technique centrale. Elle doit permettre à quelqu'un d'autre de **reproduire exactement** vos analyses.

#### Tests statistiques
- Test principal pour chaque hypothèse (Welch, Mann-Whitney, ANOVA, régression, modèle mixte…)
- Unilatéral vs bilatéral — justifié par la directionnalité de l'hypothèse
- Seuil de significativité α et s'il est corrigé (Bonferroni, FDR…)
- Taille d'effet reportée (d, η², r, OR…)

#### Logiciel & pipeline
- Langage (Python, R…), version, librairies principales avec versions
- Lier un script ou notebook en annexe ou dans un dépôt OSF/GitHub si possible

#### Gestion des données manquantes
- Définition du seuil d'exclusion (e.g., > X% → exclusion du participant)
- Stratégie : listwise deletion, imputation (mean, MICE, KNN) ?
- Analyse de sensibilité prévue ?

#### Valeurs aberrantes (outliers)
- Critère de détection : ± 3 SD, IQR × 1.5, test de Grubbs, distance de Mahalanobis…
- Action : exclusion, Winsorisation, transformation ?
- Décision prise avant ou après analyse principale ?

#### Corrections pour comparaisons multiples
- Si plusieurs hypothèses testées, déclarer la méthode de correction
- **Bonferroni** : conservatrice, adaptée si peu de tests
- **FDR (Benjamini-Hochberg)** : plus puissante pour de nombreux tests (NLP, neuroimagerie)
- Déclarer si les analyses exploratoires sont exclues des corrections

> ⚠️ Toute décision analytique non listée ici (choix de covariable, retrait d'un participant) devra être déclarée comme **exploratoire** dans le papier final.

---

### 2.7 Analyses exploratoires

Les analyses exploratoires sont légitimes et importantes — elles génèrent des hypothèses pour les études futures. Elles doivent simplement être clairement étiquetées.

- Lister les analyses supplémentaires prévues mais non confirmatives
- Indiquer qu'aucune correction pour tests multiples n'est appliquée (ou préciser laquelle)
- Inclure les analyses de sous-groupes, modélisations alternatives, visualisations

> 💡 Dans le papier, utilisez un label explicite : **« Analyses exploratoires (non pré-enregistrées) »**. Certains journaux l'exigent désormais.

---

## 3. Plateformes de pré-enregistrement

| Plateforme | Description |
|---|---|
| **OSF** (Open Science Framework) | La plus flexible. Formulaire libre ou template AsPredicted/Prereg Challenge. Timestamp vérifiable. Idéal pour les sciences cognitives et NLP. |
| **AsPredicted.org** | Formulaire en 8 questions, très guidé, rapide. Permet l'anonymisation pour la revue. Adapté aux études comportementales et cliniques. |
| **ClinicalTrials.gov** | Obligatoire pour les essais cliniques. Pas adapté aux études observationnelles en NLP. |
| **PROSPERO** | Dédié aux revues systématiques et méta-analyses. Non pertinent pour études empiriques primaires. |
| **Zenodo / GitHub** | Pour déposer le protocole et le code, en complément d'OSF. |

> 💡 Pour une étude sur les graphes de parole dans les SSD : **OSF est recommandé**. AsPredicted est un bon complément pour son formulaire guidé.

---

## 4. Quand soumettre le pré-reg ?

| Timing | Statut |
|---|---|
| Avant toute collecte de données | ✅ Idéal |
| Après collecte, avant analyse (mentionné explicitement) | ✅ Acceptable |
| Avant de regarder les résultats (corpus existant) | ✅ Acceptable en NLP/corpus |
| Après avoir vu des résultats d'analyse | ❌ Trop tard |

> ⚠️ Si les données sont déjà partiellement analysées, mentionnez-le honnêtement. Les reviewers préfèrent la transparence à un timestamp qui ne correspond pas à la réalité.

---

## 5. Faire le lien dans le papier

#### Dans la section Méthode

Ajouter une phrase du type :

> *« Cette étude a été pré-enregistrée avant l'analyse des données sur [plateforme] : [URL/DOI]. Le plan d'analyse suit le protocole pré-enregistré, à l'exception de [décrire les déviations]. »*

#### Déviations au pré-reg
- Toute déviation doit être **déclarée et justifiée** dans le papier
- Elle ne remet pas en cause la validité de l'étude — la transparence est la norme
- Distinguer : déviation dans le cadre confirmatoire vs analyse exploratoire

#### Badge Open Science
- Certains journaux accordent un badge « Preregistered » visible sur l'article
- Conditions : timestamp vérifiable, lien accessible, analyse cohérente avec le pré-reg

---

## 6. Template rapide — 8 questions AsPredicted

| Question | Contenu attendu |
|---|---|
| **Q1** — Données déjà collectées ? | Oui / Non / En cours. Si oui, précisez ce que vous avez vu. |
| **Q2** — Hypothèses principales | Liste numérotée, directionnelle si possible. |
| **Q3** — Variables dépendantes | Nom + mode de calcul + unité pour chaque VD. |
| **Q4** — Conditions / groupes | Noms des groupes, taille, randomisation éventuelle. |
| **Q5** — Analyses | Tests statistiques pour chaque hypothèse, logiciel, α. |
| **Q6** — Outliers & exclusions | Critères d'exclusion des participants et des données. |
| **Q7** — Taille d'échantillon | N total + calcul de puissance ou justification. |
| **Q8** — Autre | Covariables, analyses exploratoires, code, informations complémentaires. |

---

## 7. Erreurs fréquentes à éviter

- **Hypothèses vagues** : « on s'attend à des différences entre les groupes » n'est pas suffisant
- **Omettre le sens du test** : bilatéral vs unilatéral doit être justifié
- **Oublier la gestion des outliers** : une décision prise après les données est du HARKing analytique
- **Ne pas spécifier les covariables** : les inclure après coup biaise les résultats
- **Mélanger confirmatoire et exploratoire** : étiquetez toujours clairement
- **Ne pas référencer le pré-reg dans le papier** : c'est sa raison d'être
- **Modifier le pré-reg sans historique** : utilisez les amendements datés sur OSF

---

*Document de référence — Mars 2026*
