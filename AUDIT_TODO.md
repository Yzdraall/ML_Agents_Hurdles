# Audit du projet – ML-Agents 110m Hurdles

> Audit réalisé à partir du code (`Assets/Scripts`), de la scène `SampleScene.unity`, des prefabs, des fichiers de configuration `*.yaml`, du dossier `results/` et des `ProjectSettings`.
> Version Unity : **6000.0.66f1** · ML-Agents : **com.unity.ml-agents 4.0.2**

Légende priorité : 🔴 critique (bug / bloquant) · 🟠 important (qualité d'apprentissage) · 🟡 amélioration · ⚪ confort / hygiène

---

## 1. Ce qui marche bien

- **Concept clair et ambitieux** : deux morphologies (bipède / quadrupède) sur un même parcours de haies, comportement émergent sans animation.
- **Entraînement parallélisé** : 16 Walkers + 16 Crawlers dans la même scène → bonne utilisation de la collecte d'expérience.
- **Stack récente** : Unity 6, ML-Agents 4.0.2, nouvel Input System, export ONNX.
- **Hyperparamètres PPO raisonnables** : `normalize: true`, réseau 3×512, `gamma 0.995`, `lr 3e-4` linéaire – proches des exemples officiels Walker/Crawler.
- **Inspector bien organisé** : `[Header]`, `[Tooltip]`, récompenses et forces exposées comme paramètres → itération rapide sans recompiler.
- **Observations cohérentes avec les Behavior Parameters** : Walker = 132 obs / 15 actions, Crawler = 81 obs / 8 actions (vérifié par calcul).
- **Debug visuel** des raycasts (`Debug.DrawLine` coloré selon haie / obstacle / vide).
- **Tags constants** (`FLOORTAG`, `HURDLETAG`) plutôt que chaînes dispersées (dans les agents).
- **Runs versionnés** (`v1.3_M`, `v1.4_M`, …) et `configuration.yaml` conservé par run → reproductibilité partielle.
- **README avec GIF de démo** : bonne vitrine du projet.

---

## 2. Problèmes identifiés et améliorations

### 2.1 🔴 Bugs / incohérences critiques

| # | Problème | Où | Pourquoi c'est un problème |
|---|----------|----|----------------------------|
| B1 | Le fichier `Crawler_Agent.cs` déclare la classe `Walker_Agent`, et `Walker_Agent.cs` déclare `WalkerAgent` | [Assets/Scripts/Crawler_Agent.cs](Assets/Scripts/Crawler_Agent.cs#L9), [Assets/Scripts/Walker_Agent.cs](Assets/Scripts/Walker_Agent.cs#L9) | Unity exige que le nom d'un `MonoBehaviour` corresponde au nom du fichier. Sinon : « The associated script cannot be loaded » / composant manquant. Très source de confusion. |
| B2 | `collision.gameObject.GetComponent<GroundContact>()` teste **l'objet touché (le sol)**, pas la partie du corps | [Walker_Agent.cs](Assets/Scripts/Walker_Agent.cs#L246-L255), idem Crawler | La condition est toujours vraie → la vérification « ce n'est pas un pied » ne sert à rien. |
| B3 | `OnCollisionEnter` est placé sur l'objet racine de l'agent | [Walker_Agent.cs](Assets/Scripts/Walker_Agent.cs#L240) | Les callbacks de collision ne sont envoyés qu'au GameObject portant le `Collider`/`Rigidbody`. La racine du ragdoll n'en a (très probablement) pas → **la pénalité de haie n'est jamais appliquée**. |
| B4 | Le reset ne replace que les hanches | [Walker_Agent.cs](Assets/Scripts/Walker_Agent.cs#L90-L113) | Les autres membres restent à leur position de chute puis sont « tirés » par les joints → explosions physiques, épisodes de départ non reproductibles. |
| B5 | Les modèles ONNX ne sont pas assignés dans la scène (31 `m_Model` à `null`, 1 référence vers un GUID inexistant) | `SampleScene.unity` | La scène ne fonctionne pas en inférence « out of the box », contrairement à ce que laisse penser le README. |
| B6 | Le README indique que les modèles sont dans `results/v1.4_M/` | [README.md](README.md) | Unity ne peut référencer que des assets sous `Assets/`. Instruction incorrecte. |
| B7 | Le prefab `WalkerRagdoll.prefab` référence des scripts des exemples ML-Agents absents du projet (`JointDriveController`, `OrientationCubeController`, etc.) | `Assets/ML-Agents/Walker/Prefabs/Ragdoll/` | Composants « Missing Script » → warnings, comportement imprévisible si le prefab est réutilisé. |
| B8 | `no_hand_touch` appelle `EndEpisode()` dans `OnCollisionStay` et ignore son champ `targetTag` (utilise `"floor"` en dur) | [no_hand_touch.cs](Assets/Scripts/no_hand_touch.cs#L20-L29) | Appels multiples possibles pendant une même frame physique ; champ public inutile. Ne fonctionne que pour `WalkerAgent`, pas pour le Crawler. |
| B9 | Seuils de chute en hauteur **absolue** (`0.5f` pour Walker, `1.5f` pour Crawler) | [Walker_Agent.cs](Assets/Scripts/Walker_Agent.cs#L222), [Crawler_Agent.cs](Assets/Scripts/Crawler_Agent.cs#L214) | Dépend de l'altitude de la piste ; casse si on déplace une zone. Valeurs semblant inversées (un quadrupède est plus bas qu'un bipède). |

### 2.2 🟠 Qualité de l'apprentissage (RL)

| # | Problème | Recommandation |
|---|----------|----------------|
| R1 | Les raycasts partent de `hanches + 0.5 m` et sont horizontaux : ils passent probablement **au-dessus** des haies (≈ 1,07 m) et pivotent avec l'inclinaison du corps. | Utiliser le composant **`RayPerceptionSensorComponent3D`** (tags détectables, layer mask, angle vertical, gizmos) ou des rayons inclinés vers le bas, stabilisés en lacet uniquement (`Quaternion.Euler(0, hips.eulerAngles.y, 0)`). Ajouter un `LayerMask` pour ne pas toucher son propre corps. |
| R2 | Observations en **repère monde** (vitesse, direction cible). | Les exprimer dans un repère local stabilisé (`hips.InverseTransformDirection(...)` ou « orientation cube » comme dans l'exemple Walker officiel) → meilleure généralisation. |
| R3 | Observation directe de la prochaine haie absente. | Ajouter distance à la prochaine haie (normalisée) et sa hauteur – beaucoup plus simple à apprendre que de la déduire des rayons. |
| R4 | Le « coût énergétique » `-0.001f` est en réalité une **pénalité de temps** constante. | Ajouter un vrai coût d'effort : `-k * Σ action²`, et éventuellement pénaliser les variations brusques d'action. |
| R5 | Une seule action par joint (axe X, ±60° pour tous). | Respecter les limites réelles de chaque joint (`lowAngularXLimit`/`highAngularXLimit`), ajouter les axes Y/Z pour les articulations qui en ont (hanches, épaules) et une action de **force** par membre (comme `JointDriveController`). |
| R6 | Récompense de vitesse non bornée et sans notion d'orientation. | S'inspirer du Walker officiel : `reward = matchSpeed × lookAtTarget`, avec vitesse cible et hauteur de tête. |
| R7 | Aucun curriculum. | Utiliser `environment_parameters` + `Academy.Instance.EnvironmentParameters` : hauteur des haies 0 → 0.3 → 0.7 → 1.07 m, espacement variable. |
| R8 | Pas de randomisation (départ, espacement des haies, masse). | Domain randomization légère pour éviter l'overfitting sur un parcours fixe. |
| R9 | `time_horizon: 64` avec `gamma: 0.995` (horizon effectif ≈ 200 pas). | Passer à `time_horizon: 1000` (valeur des exemples Walker/Crawler). |
| R10 | `Fixed Timestep = 0.02` (50 Hz). | Les exemples ragdoll utilisent ~0.0133 (75 Hz) ; tester pour une physique plus stable. |
| R11 | `GroundContact` n'est pas réinitialisé au début d'épisode et passe à `false` dès qu'un seul collider sort. | Reset dans `OnEpisodeBegin`, compteur de contacts au lieu d'un booléen. Le Crawler n'a **aucun** `GroundContact` (81 obs = 0 pied) → en ajouter sur ses 4 pattes. |
| R12 | Pas de métriques métier dans TensorBoard. | `Academy.Instance.StatsRecorder.Add("Hurdles/Cleared", n)`, distance parcourue, temps au 110 m, taux d'arrivée. |

### 2.3 🟡 Architecture & qualité du code

| # | Problème | Recommandation |
|---|----------|----------------|
| C1 | `Crawler_Agent.cs` est un copier-coller de `Walker_Agent.cs` (~270 lignes dupliquées). | Créer une classe abstraite `HurdleAgentBase : Agent` (observations, raycasts, reset, récompenses communes) et deux sous-classes `WalkerHurdleAgent` / `CrawlerHurdleAgent` qui ne surchargent que le spécifique (seuils, partie « hanches »/« corps »). |
| C2 | `GetComponent<Rigidbody>()` appelé pour chaque joint à chaque décision ; `GetComponentsInChildren<Rigidbody>()` à chaque reset. | Mettre en cache dans `Initialize()`. |
| C3 | Téléportation via `transform.position` après `rb.Sleep()`. | Utiliser `rb.position`/`rb.rotation` ou `Physics.SyncTransforms()` ; stocker la pose initiale locale de **chaque** membre. |
| C4 | Valeurs magiques (`60f`, `0.5f`, `0.6f`, `0.2f`, `2.5f`, `0.005f`, `30f`…). | Les exposer en champs `[SerializeField]` ou constantes nommées. |
| C5 | Conventions de nommage hétérogènes (`no_hand_touch`, `Walker_Agent`, `WalkerAgent`). | PascalCase partout : `BodyFloorPenalty`, `WalkerHurdleAgent`… |
| C6 | Champs publics pour tout. | `[SerializeField] private` pour l'encapsulation. |
| C7 | Commentaires informels / emoji dans les logs (« ton bug », `⚠️`). | Nettoyer pour un dépôt public/portfolio. |
| C8 | Usings inutiles (`TMPro`, `Unity.VisualScripting`, `Unity.MLAgents` dans `no_hand_touch`). | Supprimer. |
| C9 | `Heuristic()` contracte tous les muscles avec Espace : inutilisable pour déboguer. | Mapper quelques touches à des groupes de joints, ou supprimer. |
| C10 | `CameraFollow` utilise `Lerp` avec un facteur fixe dans `LateUpdate` (dépend du framerate). | `Vector3.SmoothDamp` ou `1 - Mathf.Exp(-speed * Time.deltaTime)`. Option : bascule entre agents au clavier. |
| C11 | Le tag `finish` existe mais n'est pas utilisé ; l'arrivée est détectée par distance `< 2.5f`. | Trigger de ligne d'arrivée avec le tag `finish`. |

### 2.4 ⚪ Organisation du dépôt & assets

| # | Problème | Recommandation |
|---|----------|----------------|
| D1 | `results/` pèse **~1,2 Go** (513 fichiers dans `v1.4_M`) et n'est pas ignoré. | Ajouter `results/` au `.gitignore`, ne garder que les ONNX finaux dans `Assets/Models/Brains/`. |
| D2 | Pas de `.gitattributes` / Git LFS pour `.onnx`, `.onnx.data`, `.fbx`, `.gif`, `.png`. | `git lfs track "*.onnx" "*.onnx.data" "*.fbx" "*.gif" "*.png"`. |
| D3 | ONNX dupliqués à 4 endroits (`Assets/ML-Agents`, `Assets/Resources`, `Assets/TutorialInfo`, `Walker/TFModels`). | Un seul dossier `Assets/Models/Brains/` avec noms explicites (`Walker_v1.4_100M.onnx`). |
| D4 | `Assets/Resources/` contient des captures d'écran et des modèles non chargés par code. | Tout ce qui est dans `Resources` est embarqué dans le build : déplacer/supprimer. |
| D5 | Restes du template URP (`TutorialInfo/`, `Readme.asset`). | Supprimer. |
| D6 | `Timers/SampleScene_timers.json` (généré par ML-Agents). | Ajouter au `.gitignore`. |
| D7 | Noms génériques : `SampleScene`, `New Material 1..5`. | `HurdlesTraining.unity`, `HurdlesDemo.unity`, `M_Track`, `M_Hurdle`… |
| D8 | Packages inutilisés (`collab-proxy`, `visualscripting`, `multiplayer.center`, `ai.navigation`). | Les retirer du `manifest.json` (compilation plus rapide). |
| D9 | 3 fichiers YAML qui se recopient (`walker`, `crawler`, `multi`) avec `max_steps` incohérents (20M / 100M / 110M). | Un seul `config/hurdles.yaml` (ou `default_settings:` + surcharges), dans un dossier `config/`. |
| D10 | Behavior names = noms de fichiers de config (`walker_config`, `crawler_config`). | `HurdleWalker`, `HurdleCrawler`. |
| D11 | Pas de `LICENSE`. | Ajouter (MIT par ex.) + mentionner les assets ML-Agents (Apache 2.0). |

### 2.5 ⚪ Documentation

| # | Recommandation |
|---|----------------|
| DOC1 | Indiquer les versions exactes : Unity 6000.0.66f1, `com.unity.ml-agents` 4.0.2, Python (3.10.x) et `mlagents` pip correspondant. |
| DOC2 | Ajouter la section **Entraîner** : création du venv, `pip install mlagents`, commande `mlagents-learn config/hurdles.yaml --run-id=... --env=Build/... --num-envs=N --no-graphics`. |
| DOC3 | Corriger la section **Tester** (nom de la scène, emplacement réel des ONNX). |
| DOC4 | Décrire observations / actions / fonction de récompense / conditions de fin (tableau). |
| DOC5 | Ajouter courbes TensorBoard (récompense cumulée, longueur d'épisode) et tableau des runs (v1.3 → v1.6 : ce qui a changé, résultat). |
| DOC6 | Uniformiser la langue (README en anglais, code commenté en français). |

---

## 3. Plan d'action (checklist)

### Phase 1 – Remettre le projet dans un état sain 🔴
- [x] **B1** Renommer les classes pour correspondre aux fichiers (`CrawlerAgent` dans `CrawlerAgent.cs`, `WalkerAgent` dans `WalkerAgent.cs`) – faire le renommage **dans l'éditeur Unity** pour conserver les GUID des `.meta`. *(fait : fichiers + `.meta` renommés ensemble, GUID conservés ; `no_hand_touch.cs` → `BodyPartContact.cs`)*
- [x] **B2/B3** Déplacer la détection de collisions (haie / corps au sol) dans un composant posé sur **chaque membre** (fusion avec `no_hand_touch` → `BodyPartContact`) qui notifie l'agent parent. *(ajouté automatiquement par `HurdleAgentBase` sur chaque `Rigidbody`)*
- [x] **B4/C3** Sauvegarder la pose locale initiale de tous les `Rigidbody` dans `Initialize()` et la restaurer intégralement dans `OnEpisodeBegin()`.
- [x] **B8** Utiliser `OnCollisionEnter` + garde « épisode déjà terminé », utiliser `targetTag`, typer l'agent sur la classe de base.
- [x] **B9** Seuils de chute relatifs : `hips.position.y < startPosition.y - fallMargin`.
- [x] **B5/B6** Assigner les modèles ONNX finaux dans la scène et corriger le README.
- [x] **B7** Nettoyer les « Missing Script » des prefabs (ou créer des variantes propres).

### Phase 2 – Améliorer l'apprentissage 🟠
- [x] **R1** Remplacer les raycasts manuels par `RayPerceptionSensorComponent3D` (tags `hurdle`, `floor`, `finish`, angle vers le bas, layer mask). *(capteur `HurdleRays` créé au runtime sur un enfant stabilisé suivant la racine ; couches `Walker`/`Crawler` exclues ; les 32 capteurs statiques de la scène ont été retirés)*
- [x] **R2/R3** Observations en repère local + distance/hauteur de la prochaine haie.
- [x] **R4** Vrai coût d'effort sur les actions.
- [x] **R5** Respect des limites de joints, actions multi-axes et force par membre.
- [x] **R6** Récompense « vitesse × orientation » inspirée du Walker officiel.
- [x] **R9** `time_horizon: 1000` ; comparer sur un run court.
- [x] **R11** `GroundContact` : compteur + reset ; ajouter les pieds du Crawler.
- [x] **R12** Métriques custom via `StatsRecorder`.
- [x] **R7** Curriculum sur la hauteur des haies. *(`walker_hurdle_height` / `crawler_hurdle_height` dans `config/hurdles.yaml`)*
- [x] **R8** Domain randomization (position de départ, espacement des haies).
- [x] **R10** Tester `Fixed Timestep = 0.0133`. *(appliqué ; `MaxStep` 3500 → 5250 pour garder 70 s)*

### Phase 3 – Refactoring du code 🟡
- [x] **C1** Classe de base `HurdleAgentBase` + 2 sous-classes (`WalkerHurdleAgent`, `CrawlerHurdleAgent`).
- [x] **C2** Cache des composants.
- [x] **C4/C6** Champs `[SerializeField] private`, suppression des valeurs magiques.
- [x] **C5/C7/C8** Nommage, commentaires, usings.
- [x] **C9** Heuristic utile ou supprimée. *(`[`/`]` choix du joint, flèches = axes X/Y, Espace = tout fléchir)*
- [x] **C10** Caméra indépendante du framerate. *(+ Tab pour changer d'agent)*
- [x] **C11** Trigger d'arrivée avec le tag `finish`.

### Phase 4 – Dépôt & documentation ⚪
- [x] **D1/D6** Compléter le `.gitignore` (`results/`, `Assets/ML-Agents/Timers/`, `*.onnx.data` intermédiaires).
- [x] **D2** Mettre en place Git LFS. *(`.gitattributes` ; lancer `git lfs install` après `git init`)*
- [x] **D3/D4/D5** Dédupliquer les ONNX, vider `Resources/`, supprimer `TutorialInfo/`.
- [x] **D7/D10** Renommer scènes, matériaux, behaviors. *(`HurdlesTraining.unity`, `M_Hurdle_*`, `HurdleWalker`/`HurdleCrawler`)*
- [x] **D8** Nettoyer `manifest.json`.
- [x] **D9** Unifier les configs YAML dans `config/`.
- [x] **D11** Ajouter une licence.
- [x] **DOC1 → DOC6** Réécrire le README (versions, entraînement, inférence, design RL, résultats).

---

## 4. Exemples d'implémentation

### 4.1 Classe de base commune (C1)

```csharp
public abstract class HurdleAgentBase : Agent
{
    [SerializeField] protected Transform target;
    [SerializeField] protected Transform root;           // hanches ou corps
    [SerializeField] protected float fallMargin = 0.5f;  // relatif à la hauteur de départ

    struct BodyPose { public Rigidbody rb; public Vector3 pos; public Quaternion rot; }
    readonly List<BodyPose> initialPose = new();
    protected readonly List<ConfigurableJoint> joints = new();
    protected readonly List<Rigidbody> jointBodies = new();
    protected float startHeight;

    public override void Initialize()
    {
        foreach (var rb in GetComponentsInChildren<Rigidbody>())
            initialPose.Add(new BodyPose { rb = rb, pos = rb.transform.localPosition, rot = rb.transform.localRotation });

        GetComponentsInChildren(joints);
        foreach (var j in joints) jointBodies.Add(j.GetComponent<Rigidbody>());
        startHeight = root.position.y;
    }

    public override void OnEpisodeBegin()
    {
        foreach (var p in initialPose)
        {
            p.rb.transform.SetLocalPositionAndRotation(p.pos, p.rot);
            p.rb.linearVelocity = Vector3.zero;
            p.rb.angularVelocity = Vector3.zero;
        }
        Physics.SyncTransforms();
    }

    protected bool HasFallen() => root.position.y < startHeight - fallMargin;

    public void OnBodyPartTouchedFloor() { AddReward(-1f); EndEpisode(); }
    public void OnHurdleHit()            { AddReward(-0.5f); }
}
```

### 4.2 Détection de contact par membre (B2, B3, B8)

```csharp
public class BodyPartContact : MonoBehaviour
{
    [SerializeField] bool isFoot;
    HurdleAgentBase agent;

    void Awake() => agent = GetComponentInParent<HurdleAgentBase>();

    void OnCollisionEnter(Collision c)
    {
        if (c.collider.CompareTag("hurdle")) agent.OnHurdleHit();
        else if (!isFoot && c.collider.CompareTag("floor")) agent.OnBodyPartTouchedFloor();
    }
}
```

### 4.3 Configuration unifiée avec curriculum (D9, R7, R9)

```yaml
default_settings:
  trainer_type: ppo
  hyperparameters: { batch_size: 2048, buffer_size: 20480, learning_rate: 3.0e-4,
                     beta: 5.0e-3, epsilon: 0.2, lambd: 0.95, num_epoch: 3,
                     learning_rate_schedule: linear }
  network_settings: { normalize: true, hidden_units: 512, num_layers: 3 }
  reward_signals: { extrinsic: { gamma: 0.995, strength: 1.0 } }
  time_horizon: 1000
  summary_freq: 50000
  keep_checkpoints: 5

behaviors:
  HurdleWalker:  { max_steps: 100000000 }
  HurdleCrawler: { max_steps: 100000000 }

environment_parameters:
  hurdle_height:
    curriculum:
      - name: Flat
        completion_criteria: { measure: reward, behavior: HurdleWalker, threshold: 50, min_lesson_length: 100 }
        value: 0.0
      - name: Low
        completion_criteria: { measure: reward, behavior: HurdleWalker, threshold: 80, min_lesson_length: 100 }
        value: 0.4
      - name: Real
        value: 1.067
```

### 4.4 Ajouts au `.gitignore` (D1, D6)

```gitignore
# ML-Agents
/results/
/[Aa]ssets/ML-Agents/Timers/
/[Aa]ssets/ML-Agents/Timers.meta
```

---

## 5. Troisième passe : vérification croisée code / scène / prefabs

Analyse du YAML de `HurdlesTraining.unity` et des prefabs (overrides effectifs, positions des haies et des lignes d'arrivée) confrontée au code. Sauvegarde des fichiers modifiés : `.backup_audit_3/`.

### 5.1 Problèmes trouvés et corrigés

| # | Problème | Correction |
|---|----------|------------|
| P1 | `Assets/Scripts.meta` et `HurdleAgentBase.cs.meta` absents : GUID régénéré différemment à chaque clone. | `.meta` ajoutés. |
| P2 | Repère de piste orienté vers le **centre** de l'objet `finish` : axe incliné de 2,7° (Walker) et 0,9° (Crawler) par rapport au couloir. | Direction vers le point le plus proche du collider d'arrivée (perpendiculaire à la ligne). Longueur ≈ 283,7 m (Walker), 282,8 m (Crawler). |
| P3 | Une seule pièce de haie retenue par rangée (marge de 0,05 m pour le Crawler). Les pièces voisines (bord à 1–1,7 m de l'axe) restaient à pleine hauteur pendant les leçons `Flat` / `Low`. | Toutes les pièces taguées `hurdle` à moins de `laneMargin` (2 m) de l'axe sont prises et regroupées par rangée : 7 rangées par couloir, aucune pièce partagée entre deux agents. Une rangée = un décalage aléatoire, une observation, une récompense. |
| P4 | Curriculum : la fraction s'appliquait à la hauteur de la boîte de collision, en partie enfoncée sous la piste. `Low` (0,3) était entièrement plat et `Medium` (0,7) presque plat. | La fraction s'applique à la hauteur du sommet au-dessus de la piste (raycast vers le sol tagué `floor`). |
| P5 | Moteurs articulaires incohérents à cause des overrides de scène. `thighR` était en mode X & YZ : son action de force était ignorée, son axe Y n'avait aucun moteur (son action Y restait sans effet) et l'amortissement X était 10× plus fort. `footL` était en mode force alors que les autres articulations étaient en accélération. | L'agent impose `Slerp` et un drive en accélération sur toutes les articulations. Les 1 296 overrides de drives ont été supprimés de la scène. |
| P6 | Détection automatique de la racine : `transform.Find("Hips")` ne trouvait jamais `hips`. | Premier `Rigidbody` de la hiérarchie. |
| P7 | `Track Progress` < 1 pour un agent arrivé par contact avant la ligne. | Vaut 1 quand l'arrivée est atteinte. |
| P8 | Prefab Walker : sous-objets inactifs `FootRays` (4 capteurs de rayons) et `OrientationCube` repris de l'exemple ML-Agents. Matériaux manquants sur 23 renderers (masqués par 368 overrides de scène). | Sous-objets supprimés. `M_Walker_Black` / `M_Walker_Red` / `M_Walker_White` sont assignés dans le prefab. |
| P9 | Prefab Crawler : script manquant (`PhysicsBodySensor` du package `ml-agents.extensions`, non installé) et deux prefabs imbriqués manquants (`OrientationCube`, `DirectionIndicator`). Matériaux manquants sur 17 renderers. | Supprimés. `crawler_skin` / `crawler_feet` sont assignés dans le prefab. |
| P10 | Scène : 640 overrides de matériaux, overrides de Behavior Parameters identiques au prefab (dont `m_Model` à `None`, qui empêchait l'assignation du modèle au niveau du prefab), 48 entrées `m_RemovedComponents` vers des composants inexistants, 32 overrides de layer orphelins. | Supprimés. `m_InferenceDevice` est conservé. |
| P11 | Scène : 48 murs latéraux nommés `finish (n)` (non tagués) et 16 copies `floor (4)` parasites, enfants des sols et placées à ≈ 1,5 km de la piste. | Murs renommés `wall (n)`, copies supprimées. |
| P12 | Matériaux du Walker nommés `New Material`, `New Material 1` et `New Material 2`. | Renommés `M_Walker_Red`, `M_Walker_Black` et `M_Walker_White` (GUID conservés). |
| P13 | `ML_Agents_110m_Hurdles.slnx` (solution générée) non ignoré. | `*.slnx` ajouté au `.gitignore`. |

### 5.2 Points ouverts

- [ ] Ré-entraîner : les drives, le curriculum et les rangées de haies changent la dynamique et la tâche.
- [ ] Tester dans l'éditeur (Play) : aucune compilation n'a pu être faite hors de Unity.
- [ ] Rangée rouge (≈ 241 m) : pièces de 4,6 m de haut (sommet à 5,3 m au-dessus de la piste), certaines flottant à 0,69 m au-dessus de la piste, une autre posée au sol. Elle est sans doute infranchissable à hauteur réelle, même pour le Walker à l'échelle 3,5. Revoir le level design.
- [ ] Les haies posées à la main se décalent latéralement d'environ ±2 m d'une rangée à l'autre. `laneMargin` couvre ce décalage.
- [ ] Les composants d'agent de la scène contiennent encore des champs obsolètes (`drawRaycastsInScene`, `raycastDistance`…). Unity les supprimera à la prochaine sauvegarde de la scène.
