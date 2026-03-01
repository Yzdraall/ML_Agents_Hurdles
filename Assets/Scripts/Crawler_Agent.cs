using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
// Nous utilisons le nouveau système d'input pour éviter tes erreurs précédentes
using UnityEngine.InputSystem; 

public class Walker_Agent : Agent
{
    [Header("--- OBJECTIFS & CORPS ---")]
    [Tooltip("La ligne d'arrivée ou le cube cible à atteindre")]
    public Transform target;
    
    [Tooltip("GLISSEZ ICI LES HANCHES (Hips) DU ROBOT ! C'est crucial pour détecter la chute.")]
    public Transform hips; 

    [Header("--- REGLAGES MUSCLES ---")]
    [Tooltip("Force des muscles (Spring). Augmentez si le robot s'effondre comme une crêpe.")]
    public float muscleStrength = 2000f; 
    [Tooltip("Résistance des muscles (Damper). Pour éviter qu'il tremble.")]
    public float muscleDamper = 100f;
    [Tooltip("Force maximale absolue que le muscle peut appliquer.")]
    public float maxMuscleForce = 10000f;

    [Header("--- PARAMETRES DE VISION (Raycasts) ---")]
    public bool drawRaycastsInScene = true;
    public float raycastDistance = 20f;
    [Tooltip("Hauteur d'où partent les rayons (depuis les hanches)")]
    public float raycastHeightOffset = 0.5f; 

    [Header("--- RECOMPENSES & PUNITIONS ---")]
    public float targetReachedReward = 10f;
    public float fallPenalty = -1.0f;
    public float hurdleHitPenalty = -0.5f;
    public float moveSpeedRewardFactor = 0.1f;
    
    // --- VARIABLES INTERNES (Ne pas toucher) ---
    private Rigidbody rootRb; // Le rigidbody des hanches
    private List<ConfigurableJoint> joints; // Liste de tous les muscles
    private List<GroundContact> feet; // Liste des pieds (pour le saut)
    private Vector3 startPosition; // Position de spawn
    private Quaternion startRotation; // Rotation de spawn
    private const string FLOORTAG = "floor";
    private const string HURDLETAG = "hurdle";

    // Initialisation : Se lance une seule fois au début
    public override void Initialize()
    {
        // 1. Configuration des Hanches (Le centre du robot)
        if (hips == null)
        {
            // Tentative de récupération automatique si tu as oublié de le glisser
            hips = transform.Find("Hips"); 
            if (hips == null) hips = transform.GetComponentInChildren<Rigidbody>().transform;
        }
        
        rootRb = hips.GetComponent<Rigidbody>();
        
        // Sauvegarde de la position de départ pour le respawn
        startPosition = hips.position;
        startRotation = hips.rotation;

        // 2. Récupération des pieds (Doivent avoir le script GroundContact)
        feet = new List<GroundContact>(GetComponentsInChildren<GroundContact>());
        
        // 3. Récupération et Configuration automatique des Joints (Muscles)
        joints = new List<ConfigurableJoint>(GetComponentsInChildren<ConfigurableJoint>());
        
        foreach (var joint in joints)
        {
            // On configure les muscles pour qu'ils soient puissants
            var driveX = joint.angularXDrive;
            driveX.positionSpring = muscleStrength;
            driveX.positionDamper = muscleDamper;
            driveX.maximumForce = maxMuscleForce;
            joint.angularXDrive = driveX;

            var driveYZ = joint.angularYZDrive;
            driveYZ.positionSpring = muscleStrength;
            driveYZ.positionDamper = muscleDamper;
            driveYZ.maximumForce = maxMuscleForce;
            joint.angularYZDrive = driveYZ;
        }
    }

    // Reset : Se lance à chaque fois que le robot meurt ou gagne
    public override void OnEpisodeBegin()
    {
        // 1. On coupe toute la physique pour téléporter le robot
        foreach (var rb in GetComponentsInChildren<Rigidbody>())
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.Sleep();
        }

        hips.position = new Vector3(startPosition.x, startPosition.y + 0.5f, startPosition.z);
        hips.rotation = startRotation;

        // 3. On remet les membres en position neutre
        foreach (var joint in joints)
        {
            joint.targetRotation = Quaternion.Euler(30f, 0f, 0f);
        }
        
    }

    // Observations : Ce que le robot "voit" et "ressent"
    public override void CollectObservations(VectorSensor sensor)
    {
        
        // Direction vers la cible
        Vector3 dirToTarget = (target.position - hips.position).normalized;
        sensor.AddObservation(dirToTarget); 

        // Orientation du corps
        sensor.AddObservation(hips.forward);
        sensor.AddObservation(hips.up);

        // Vitesse
        sensor.AddObservation(rootRb.linearVelocity);
        sensor.AddObservation(rootRb.angularVelocity);

        // Position des membres (Angles et Vitesses)
        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.transform.localRotation);
            sensor.AddObservation(joint.GetComponent<Rigidbody>().angularVelocity);
        }

        // Contact des pieds (Saut)
        foreach (var foot in feet)
        {
            sensor.AddObservation(foot.IsTouchingGround ? 1f : 0f);
        }

        // --- VISION (Raycasts pour voir les Haies) ---
        // On lance 5 rayons devant lui pour détecter les obstacles
        float rayAngleStep = 10f; // Degrés entre chaque rayon
        Vector3 rayOrigin = hips.position + Vector3.up * raycastHeightOffset;
        
        for (int i = -2; i <= 2; i++) // De -20 à +20 degrés
        {
            Vector3 rayDirection = Quaternion.Euler(0, i * rayAngleStep, 0) * hips.forward;
            RaycastHit hit;
            
            if (Physics.Raycast(rayOrigin, rayDirection, out hit, raycastDistance))
            {
                // On observe la distance de l'obstacle
                sensor.AddObservation(hit.distance / raycastDistance);
                
                // On observe si c'est une haie ou un mur
                bool isHurdle = hit.collider.CompareTag(HURDLETAG);
                sensor.AddObservation(isHurdle ? 1f : 0f);

                if (drawRaycastsInScene) Debug.DrawLine(rayOrigin, hit.point, isHurdle ? Color.red : Color.blue);
            }
            else
            {
                // Rien en vue
                sensor.AddObservation(1f); 
                sensor.AddObservation(0f);
                if (drawRaycastsInScene) Debug.DrawRay(rayOrigin, rayDirection * raycastDistance, Color.green);
            }
        }
    }

    // Action : Le cerveau envoie des ordres aux muscles
    public override void OnActionReceived(ActionBuffers actions)
    {
        var act = actions.ContinuousActions;
        
        // 1. Appliquer les forces sur les joints
        int i = 0;
        foreach (var joint in joints)
        {
            if (i >= act.Length) break;

            // Le réseau de neurones donne un chiffre entre -1 et 1
            // On le convertit en rotation (ex: -60 à +60 degrés)
            float targetX = act[i++] * 60f;
            float targetY = 0f; 
            float targetZ = 0f;

            // Si on veut plus de liberté (attention, plus difficile à apprendre)
            // if (i < act.Length) targetZ = act[i++] * 15f; 

            joint.targetRotation = Quaternion.Euler(targetX, targetY, targetZ);
        }

        // 2. Calcul des Récompenses (REWARD FUNCTION)

        // A. Bonus pour la vitesse vers la cible (Uniquement s'il est debout !)
        float speedTowardsTarget = Vector3.Dot(rootRb.linearVelocity, (target.position - hips.position).normalized);
        float isUpright = Vector3.Dot(hips.up, Vector3.up); // 1 = Debout, 0 = Couché
        
        if (speedTowardsTarget > 0 && isUpright > 0.5f)
        {
            AddReward(speedTowardsTarget * moveSpeedRewardFactor);
        }

        // B. Bonus pour rester debout
        if (isUpright > 0.6f) AddReward(0.005f);

        // C. Coût énergétique (Pour éviter les mouvements épileptiques inutiles)
        AddReward(-0.001f);

        // 3. CONDITIONS DE MORT (TERMINAL STATES)

        if (hips.position.y < 1.5f) // Ajuste selon la taille de tes jambes
        {
            AddReward(fallPenalty);
            EndEpisode();
            return;
        }

        // Mort par "Tête en bas" (Si le robot est totalement retourné)
        if (isUpright < 0.2f)
        {
            AddReward(fallPenalty);
            EndEpisode();
            return;
        }

        // Victoire : Cible atteinte
        float distanceToTarget = Vector3.Distance(hips.position, target.position);
        if (distanceToTarget < 2.5f)
        {
            AddReward(targetReachedReward);
            EndEpisode();
        }
    }

    // Gestion des collisions physiques
    void OnCollisionEnter(Collision collision)
    {
        // Si on tape une haie violemment
        if (collision.collider.CompareTag(HURDLETAG))
        {
            AddReward(hurdleHitPenalty);
        }

        // Sécurité supplémentaire : Si le CORPS (pas les pieds) touche le sol
        if (collision.collider.CompareTag(FLOORTAG))
        {
            // On vérifie que ce n'est pas un pied qui a touché
            if (collision.gameObject.GetComponent<GroundContact>() == null)
            {
                AddReward(fallPenalty);
                EndEpisode();
            }
        }
    }

    // Mode Manuel (Pour tester avec le clavier)
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // Reset
        for (int k = 0; k < continuousActionsOut.Length; k++) continuousActionsOut[k] = 0;

        // Exemple simple : Appuie sur Espace pour tout contracter
        if (Keyboard.current != null && Keyboard.current.spaceKey.isPressed)
        {
            for (int k = 0; k < continuousActionsOut.Length; k++)
            {
                continuousActionsOut[k] = 1f; 
            }
        }
    }
}