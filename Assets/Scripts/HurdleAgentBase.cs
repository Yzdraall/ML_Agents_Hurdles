using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;

// Shared logic for the hurdle agents (bipedal walker and quadruped crawler):
// zero-GC flat-array body/muscle caching, kinematic hurdle rows (no static BVH rebuilds),
// dirty-checked multi-axis slerp actuation, mass-weighted CoM velocity, anti-stall, and curriculum.
[DefaultExecutionOrder(-100)]
public abstract class HurdleAgentBase : Agent
{
    public const string HurdleTag = "hurdle";
    public const string FloorTag = "floor";
    public const string FinishTag = "finish";

    // Physics layers holding agent bodies: excluded from the ray perception.
    static readonly string[] AgentLayerNames = { "Walker", "Crawler" };

    // Root velocity (3) + root angular velocity (3) + root forward (3) + root up (3)
    // + root height (1) + track progress (1) + lateral lane offset (1) + next hurdle distance & height (2).
    const int CommonObservationCount = 17;
    // Local rotation (4) + angular velocity relative to the root (3) + current strength (1).
    const int ObservationsPerMuscle = 8;
    const float MinHurdleHeightScale = 0.01f;
    const float TrailingHurdleClearance = 1.0f;
    const int SpawnSettleSteps = 30;
    const string RaySensorObjectName = "HurdleRaySensor";

    static readonly FieldInfo RequesterAgentField =
        typeof(DecisionRequester).GetField("m_Agent", BindingFlags.Instance | BindingFlags.NonPublic);
    static readonly RaycastHit[] FloorHitsBuffer = new RaycastHit[32];
    static readonly Comparison<Hurdle> CompareStartAlong = (a, b) => a.startAlong.CompareTo(b.startAlong);
    static readonly Comparison<Hurdle> CompareAlong = (a, b) => a.along.CompareTo(b.along);
    static bool allLanesAssigned;

    [Header("Track")]
    [Tooltip("Finish line of this agent's lane. The track runs straight to the nearest point of its collider.")]
    [SerializeField] Transform target;
    [Tooltip("Root body part (hips / body). Auto-detected (first rigidbody) if empty.")]
    [FormerlySerializedAs("hips")]
    [SerializeField] Transform root;
    [Tooltip("A tagged hurdle belongs to this lane when its span comes within this lateral distance (m) of the lane centre line.")]
    [SerializeField] float laneMargin = 2f;
    [Tooltip("Maximum lateral drift (m) from lane centre before the episode ends as out-of-lane.")]
    [SerializeField] float maxLaneDrift = 2.2f;
    [Tooltip("Distance (m) used to normalise the next-hurdle distance observation.")]
    [SerializeField] float hurdleObservationRange = 20f;
    [Tooltip("Height (m) used to normalise the height observations.")]
    [SerializeField] float heightObservationScale = 5f;

    [Header("Muscles")]
    [Tooltip("Slerp drive spring (acceleration drive: independent of the body-part masses).")]
    [SerializeField] float muscleStrength = 2000f;
    [Tooltip("Slerp drive damper (acceleration drive).")]
    [SerializeField] float muscleDamper = 100f;
    [Tooltip("Slerp drive maximum force when the strength action is at its maximum.")]
    [SerializeField] float maxMuscleForce = 10000f;

    [Header("Ray perception")]
    [SerializeField, Range(0, 10)] int raysPerDirection = 5;
    [SerializeField, Range(0f, 90f)] float maxRayDegrees = 45f;
    [SerializeField] float rayLength = 20f;
    [Tooltip("Ray start height relative to the root body part.")]
    [SerializeField, Range(-10f, 10f)] float rayStartVerticalOffset = 0.7f;
    [Tooltip("Ray end height relative to the root body part: lower than the start to tilt the rays towards the track.")]
    [SerializeField, Range(-10f, 10f)] float rayEndVerticalOffset = -0.6f;
    [SerializeField, Range(0f, 1f)] float raySphereRadius = 0.5f;

    [Header("Rewards")]
    [Tooltip("Target forward sprint speed (m/s). Faster speeds are not penalised.")]
    [SerializeField] float sprintTargetSpeed = 12f;
    [Tooltip("Total reward distributed strictly along new forward track progress (un-farmable).")]
    [SerializeField] float progressRewardScale = 15f;
    [Tooltip("Per-step shaping reward for forward velocity aligned with the lane.")]
    [SerializeField] float velocityShapingScale = 0.003f;
    [Tooltip("Small per-step time penalty to encourage fast finishes instead of stalling.")]
    [SerializeField] float timePenalty = 0.0015f;
    [Tooltip("Per-step quadratic penalty for drifting laterally away from the lane centre.")]
    [SerializeField] float laneCenterPenalty = 0.002f;
    [Tooltip("Weight of the effort cost: mean of the squared actions.")]
    [SerializeField] float effortPenalty = 0.002f;
    [Tooltip("Weight of the action-smoothness cost: mean of squared action deltas between steps.")]
    [SerializeField] float smoothnessPenalty = 0.001f;
    [SerializeField] float hurdleClearedReward = 0.5f;
    [SerializeField] float targetReachedReward = 10f;
    [Tooltip("Extra bonus at finish scaled by remaining episode time (rewards faster times).")]
    [SerializeField] float finishSpeedBonus = 10f;
    [SerializeField] float fallPenalty = -1f;
    [SerializeField] float hurdleHitPenalty = -0.5f;

    [Header("Termination")]
    [Tooltip("Episode ends when dot(root.up, world up) drops below this value.")]
    [SerializeField, Range(-1f, 1f)] float uprightThreshold = 0.15f;
    [Tooltip("Maximum physics steps without meaningful forward progress before ending the episode (0 = disabled).")]
    [SerializeField] int maxStallSteps = 300;
    [Tooltip("Minimum forward progress (m) required to reset the stall timer.")]
    [SerializeField] float minProgressAdvance = 0.25f;

    [Header("Curriculum & randomisation")]
    [Tooltip("Random shift (m) of each hurdle along the track at every episode.")]
    [SerializeField] float hurdleSpacingJitter = 1f;
    [Tooltip("Random lateral shift (m) of the start position.")]
    [SerializeField] float startLateralJitter = 0.3f;
    [Tooltip("Random yaw (degrees) of the start pose.")]
    [SerializeField] float startYawJitter = 10f;

    struct BodyPart
    {
        public Rigidbody rb;
        public Transform tr;
        public Vector3 startPosition;
        public Quaternion startRotation;
        public float mass;
    }

    struct Muscle
    {
        public ConfigurableJoint joint;
        public Transform tr;
        public Rigidbody rb;
        public bool x, y, z;
        public float lowX, highX, limY, limZ;
        public float strength;
        public float lastX, lastY, lastZ, lastStrength;
        public int ActionCount => (x ? 1 : 0) + (y ? 1 : 0) + (z ? 1 : 0) + 1;
    }

    // A hurdle row of the lane: kinematic pieces standing at the same distance along the track.
    sealed class Hurdle
    {
        public HurdlePiece[] pieces;
        public float startAlong;
        public float halfDepth;
        public float along;
        public float top = float.NegativeInfinity;
        public bool active = true;
    }

    struct HurdlePiece
    {
        public Transform body;
        public Rigidbody kinematicRb;
        public GameObject go;
        public Vector3 startPosition;
        public float startTop;
        public float height;
    }

    BodyPart[] bodyParts = Array.Empty<BodyPart>();
    Muscle[] muscles = Array.Empty<Muscle>();
    GroundContact[] feet = Array.Empty<GroundContact>();
    Hurdle[] hurdles = Array.Empty<Hurdle>();
    readonly Dictionary<Collider, Hurdle> colliderToHurdle = new Dictionary<Collider, Hurdle>();

    Rigidbody rootBody;
    Vector3 rootStartPosition;
    Vector3 rootLocalUp;
    float trackFloorY;
    bool hasTrackFloor;
    Vector3 trackDirection;
    Vector3 trackRight;
    Quaternion toTrackFrame;
    Quaternion rayLookRotation;
    float trackLength;
    float invTotalMass = 1f;
    int actionCount;
    float[] previousActions;
    bool hasPreviousActions;

    string behaviorName;
    string statHurdlesCleared;
    string statHurdleHits;
    string statTrackProgress;
    string statFinishRate;
    string statStallRate;
    string statFinishTime;

    BehaviorParameters behaviorParams;
    Transform raySensorTransform;
    bool bodyCached;
    bool trackGeometryReady;
    bool hurdlesCollected;
    DecisionRequester decisionRequester;

    // Episode state.
    readonly HashSet<Hurdle> hitHurdleRows = new HashSet<Hurdle>();
    readonly HashSet<Collider> hitForeignHurdles = new HashSet<Collider>();
    bool episodeEnding;
    int spawnGraceSteps;
    int nextHurdleIndex;
    int hurdlesCleared;
    int hurdleHits;
    float maxProgress;
    float stallAnchorProgress;
    int stallAnchorStep;
    bool finished;
    bool stalled;
    int episodeSteps;

    // Heuristic state.
    int heuristicMuscle;
    bool previousKeyState;
    bool nextKeyState;

    public Transform Root => root;
    public string BehaviorName => behaviorName;
    public float CurrentSpeed => rootBody != null ? rootBody.linearVelocity.magnitude : 0f;
    public float TrackProgress01 => trackLength > 0.01f ? Mathf.Clamp01(Progress / trackLength) : 0f;
    public int HurdlesClearedCount => hurdlesCleared;
    public float EpisodeTimeSeconds => (episodeSteps > 0 ? episodeSteps : StepCount) * Time.fixedDeltaTime;

    protected abstract float FallMargin { get; }
    protected abstract string BodyLayerName { get; }
    // Environment parameter (config/hurdles.yaml) scaling the hurdle height: 0 = flat track, 1 = real height.
    protected abstract string HurdleHeightParameter { get; }
    protected virtual float DefaultHurdleHeight => 1f;
    protected virtual float MinSpawnClearance => 0.02f;
    protected virtual float MinHeightAboveFloor => 0.5f;
    protected virtual Quaternion NeutralJointRotation => Quaternion.identity;
    protected abstract bool IsFootPart(Rigidbody rb);
    protected virtual bool CanTouchFloor(Rigidbody rb) => IsFootPart(rb);
    protected virtual bool ShouldEndOnFloorContact() => true;

    protected bool IsUpright => Vector3.Dot(root.rotation * rootLocalUp, Vector3.up) >= uprightThreshold;

    protected bool AnyFootTouchingGround
    {
        get
        {
            for (int i = 0; i < feet.Length; i++)
            {
                if (feet[i].IsTouchingGround) return true;
            }
            return false;
        }
    }

    float Progress => Vector3.Dot(root.position - rootStartPosition, trackDirection);
    float LateralOffset => Vector3.Dot(root.position - rootStartPosition, trackRight);

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStaticState() => allLanesAssigned = false;

    protected override void OnEnable()
    {
        RemoveStrayBaseAgents();
        EnsureDecisionRequester();

        // Body parts, track frame and ray sensor must exist before Agent.OnEnable creates sensors & policy.
        CacheBodyParts();
        InitTrackGeometry();
        EnsureRaySensor();
        SyncBrainParameters();

        base.OnEnable();
        BindDecisionRequester();

        if (Academy.Instance.IsCommunicatorOn && !Application.isEditor)
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
        }
    }

    void Start()
    {
        BindDecisionRequester();
        AssignAllLanesOnce();
    }

    void RemoveStrayBaseAgents()
    {
        foreach (var other in GetComponents<Agent>())
        {
            if (other != this && other.GetType() == typeof(Agent))
            {
                other.enabled = false;
                Destroy(other);
            }
        }
    }

    void EnsureDecisionRequester()
    {
        if (decisionRequester == null && !TryGetComponent(out decisionRequester))
        {
            decisionRequester = gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 5;
        }
        decisionRequester.TakeActionsBetweenDecisions = true;
    }

    void BindDecisionRequester()
    {
        EnsureDecisionRequester();
        if (decisionRequester.Agent != this) RequesterAgentField?.SetValue(decisionRequester, this);
    }

    public override void Initialize()
    {
        CacheBodyParts();
        InitTrackGeometry();
        EnsureRaySensor();
        if (MaxStep == 0) MaxStep = 5250;
    }

    void InitTrackGeometry()
    {
        if (trackGeometryReady) return;
        trackGeometryReady = true;

        if (target == null)
        {
            Debug.LogError($"{name}: 'target' is not assigned in the Inspector. Using root forward as track direction.", this);
            Vector3 forward = Vector3.ProjectOnPlane(root.forward, Vector3.up);
            trackDirection = forward.sqrMagnitude > 1e-4f ? forward.normalized : Vector3.forward;
            trackLength = 110f;
        }
        else
        {
            Vector3 toFinish = Vector3.ProjectOnPlane(ClosestFinishPoint() - rootStartPosition, Vector3.up);
            trackDirection = toFinish.sqrMagnitude > 1e-4f ? toFinish.normalized : Vector3.forward;
            trackLength = Mathf.Max(0.01f, toFinish.magnitude);
        }

        trackRight = Vector3.Cross(Vector3.up, trackDirection);
        rayLookRotation = Quaternion.LookRotation(trackDirection, Vector3.up);
        toTrackFrame = Quaternion.Inverse(rayLookRotation);
    }

    Vector3 ClosestFinishPoint()
    {
        if (!target.TryGetComponent(out Collider finish)) return target.position;
        return finish is MeshCollider { convex: false }
            ? finish.bounds.ClosestPoint(rootStartPosition)
            : finish.ClosestPoint(rootStartPosition);
    }

    void CacheBodyParts()
    {
        if (bodyCached) return;
        bodyCached = true;

        var rbs = GetComponentsInChildren<Rigidbody>();
        if (root == null)
        {
            for (int i = 0; i < rbs.Length; i++)
            {
                string n = rbs[i].name.ToLowerInvariant();
                if (n == "body" || n == "hips" || n == "torso" || n == "pelvis")
                {
                    root = rbs[i].transform;
                    break;
                }
            }
            if (root == null && rbs.Length > 0) root = rbs[0].transform;
        }

        rootBody = root.GetComponent<Rigidbody>();
        rootStartPosition = root.position;
        rootLocalUp = Quaternion.Inverse(root.rotation) * Vector3.up;

        int bodyLayer = LayerMask.NameToLayer(BodyLayerName);
        if (bodyLayer >= 0)
        {
            foreach (var t in GetComponentsInChildren<Transform>(true)) t.gameObject.layer = bodyLayer;
            Physics.IgnoreLayerCollision(bodyLayer, bodyLayer, true);
        }
        else
        {
            Debug.LogWarning($"{name}: physics layer '{BodyLayerName}' is missing, the rays may hit the agent's own body.", this);
        }

        bodyParts = new BodyPart[rbs.Length];
        float totalMass = 0f;
        for (int i = 0; i < rbs.Length; i++)
        {
            var rb = rbs[i];
            var tr = rb.transform;
            float m = Mathf.Max(0.001f, rb.mass);
            totalMass += m;

            bodyParts[i] = new BodyPart
            {
                rb = rb,
                tr = tr,
                startPosition = tr.position,
                startRotation = tr.rotation,
                mass = m,
            };

            bool isFoot = IsFootPart(rb);
            if (isFoot && !rb.TryGetComponent(out GroundContact _)) rb.gameObject.AddComponent<GroundContact>();

            if (!rb.TryGetComponent(out BodyPartContact contact))
            {
                contact = rb.gameObject.AddComponent<BodyPartContact>();
            }
            contact.CanTouchFloor = CanTouchFloor(rb);
            contact.Agent = this;
        }
        invTotalMass = totalMass > 0f ? 1f / totalMass : 1f;

        float floorY = FloorHeightBelow(root.position + Vector3.up * 5f);
        hasTrackFloor = !float.IsNegativeInfinity(floorY);
        if (hasTrackFloor)
        {
            trackFloorY = floorY;
            float minColliderY = float.PositiveInfinity;
            foreach (var col in GetComponentsInChildren<Collider>())
            {
                if (col.enabled && !col.isTrigger)
                    minColliderY = Mathf.Min(minColliderY, col.bounds.min.y);
            }
            float targetMinY = floorY + MinSpawnClearance;
            if (minColliderY < targetMinY)
            {
                Vector3 lift = Vector3.up * (targetMinY - minColliderY);
                rootStartPosition += lift;
                for (int i = 0; i < bodyParts.Length; i++) bodyParts[i].startPosition += lift;
            }
        }

        feet = GetComponentsInChildren<GroundContact>();

        var joints = GetComponentsInChildren<ConfigurableJoint>();
        var muscleList = new List<Muscle>(joints.Length);
        actionCount = 0;
        for (int i = 0; i < joints.Length; i++)
        {
            var joint = joints[i];
            bool x = joint.angularXMotion != ConfigurableJointMotion.Locked;
            bool y = joint.angularYMotion != ConfigurableJointMotion.Locked;
            bool z = joint.angularZMotion != ConfigurableJointMotion.Locked;
            if (!x && !y && !z) continue;

            joint.rotationDriveMode = RotationDriveMode.Slerp;
            var muscle = new Muscle
            {
                joint = joint,
                tr = joint.transform,
                rb = joint.GetComponent<Rigidbody>(),
                x = x,
                y = y,
                z = z,
                lowX = joint.lowAngularXLimit.limit,
                highX = joint.highAngularXLimit.limit,
                limY = joint.angularYLimit.limit,
                limZ = joint.angularZLimit.limit,
                lastX = float.NaN,
                lastY = float.NaN,
                lastZ = float.NaN,
                lastStrength = float.NaN,
            };
            ApplyDriveStrength(ref muscle, 1f);
            muscleList.Add(muscle);
            actionCount += muscle.ActionCount;
        }
        muscles = muscleList.ToArray();
        previousActions = new float[actionCount];
    }

    void SyncBrainParameters()
    {
        if (!TryGetComponent(out behaviorParams)) return;
        SetBehaviorName(behaviorParams.BehaviorName);

        var brain = behaviorParams.BrainParameters;
        int observationCount = CommonObservationCount + muscles.Length * ObservationsPerMuscle + feet.Length;
        if (brain.VectorObservationSize == observationCount
            && brain.ActionSpec.NumContinuousActions == actionCount
            && brain.ActionSpec.NumDiscreteActions == 0)
        {
            return;
        }

        Debug.LogWarning($"{name}: Behavior Parameters set to {observationCount} observations / {actionCount} continuous actions " +
                         $"(was {brain.VectorObservationSize} / {brain.ActionSpec.NumContinuousActions}). Update the prefab to match.", this);
        brain.VectorObservationSize = observationCount;
        brain.ActionSpec = ActionSpec.MakeContinuous(actionCount);
    }

    void SetBehaviorName(string nameStr)
    {
        if (behaviorName == nameStr && statHurdlesCleared != null) return;
        behaviorName = nameStr;
        statHurdlesCleared = $"{behaviorName}/Hurdles Cleared";
        statHurdleHits = $"{behaviorName}/Hurdle Hits";
        statTrackProgress = $"{behaviorName}/Track Progress";
        statFinishRate = $"{behaviorName}/Finish Rate";
        statStallRate = $"{behaviorName}/Stall Rate";
        statFinishTime = $"{behaviorName}/Finish Time (s)";
    }

    // Single-pass static builder: assigns all 560 hurdles across all 32 lanes once at startup
    // and attaches kinematic Rigidbodies so episode hurdle jitter never triggers PhysX Static BVH rebuilds.
    static void AssignAllLanesOnce()
    {
        if (allLanesAssigned) return;
        allLanesAssigned = true;

        var allAgents = FindObjectsByType<HurdleAgentBase>(FindObjectsSortMode.None);
        if (allAgents.Length == 0) return;

        var tempRows = new Dictionary<HurdleAgentBase, List<(Hurdle row, List<HurdlePiece> pieces)>>(allAgents.Length);
        for (int a = 0; a < allAgents.Length; a++)
        {
            var agent = allAgents[a];
            agent.CacheBodyParts();
            agent.InitTrackGeometry();
            agent.hurdlesCollected = true;
            tempRows[agent] = new List<(Hurdle, List<HurdlePiece>)>(8);
        }

        var hurdleObjects = GameObject.FindGameObjectsWithTag(HurdleTag);
        for (int h = 0; h < hurdleObjects.Length; h++)
        {
            var go = hurdleObjects[h];
            if (!go.TryGetComponent(out Collider col)) col = go.GetComponentInChildren<Collider>();
            if (col == null) continue;

            Bounds bounds = col.bounds;
            HurdleAgentBase bestOwner = null;
            float bestLateral = float.PositiveInfinity;
            float bestAlong = 0f;
            int bestId = int.MaxValue;

            for (int a = 0; a < allAgents.Length; a++)
            {
                var candidate = allAgents[a];
                if (candidate.target == null) continue;

                Vector3 offset = bounds.center - candidate.rootStartPosition;
                float along = Vector3.Dot(offset, candidate.trackDirection);
                if (along <= 0f || along >= candidate.trackLength) continue;

                float lateral = Mathf.Abs(Vector3.Dot(offset, candidate.trackRight));
                if (lateral - Extent(bounds, candidate.trackRight) > candidate.laneMargin) continue;

                int id = candidate.GetInstanceID();
                if (lateral < bestLateral - 1e-3f || (Mathf.Abs(lateral - bestLateral) <= 1e-3f && id < bestId))
                {
                    bestOwner = candidate;
                    bestLateral = lateral;
                    bestAlong = along;
                    bestId = id;
                }
            }

            if (bestOwner == null) continue;

            // Make movable hurdle pieces kinematic so episode shifts update dynamic pruning in O(1) instead of rebuilding Static BVH.
            if (!go.TryGetComponent(out Rigidbody kRb))
            {
                kRb = go.AddComponent<Rigidbody>();
            }
            kRb.isKinematic = true;
            kRb.useGravity = false;
            kRb.interpolation = RigidbodyInterpolation.None;

            float halfDepth = Extent(bounds, bestOwner.trackDirection);
            float near = bestAlong - halfDepth;
            float far = bestAlong + halfDepth;

            var ownerRows = tempRows[bestOwner];
            int matchIdx = -1;
            for (int r = 0; r < ownerRows.Count; r++)
            {
                var existing = ownerRows[r].row;
                if (near < existing.startAlong + existing.halfDepth && far > existing.startAlong - existing.halfDepth)
                {
                    matchIdx = r;
                    break;
                }
            }

            Hurdle hurdle;
            List<HurdlePiece> pieceList;
            if (matchIdx < 0)
            {
                hurdle = new Hurdle();
                pieceList = new List<HurdlePiece>(2);
                ownerRows.Add((hurdle, pieceList));
            }
            else
            {
                (hurdle, pieceList) = ownerRows[matchIdx];
                near = Mathf.Min(near, hurdle.startAlong - hurdle.halfDepth);
                far = Mathf.Max(far, hurdle.startAlong + hurdle.halfDepth);
            }

            hurdle.startAlong = hurdle.along = (near + far) * 0.5f;
            hurdle.halfDepth = (far - near) * 0.5f;
            hurdle.top = Mathf.Max(hurdle.top, bounds.max.y);

            float floor = FloorHeightBelow(new Vector3(bounds.center.x, bounds.max.y, bounds.center.z));
            float surface = float.IsNegativeInfinity(floor) ? bounds.min.y : floor;
            pieceList.Add(new HurdlePiece
            {
                body = go.transform,
                kinematicRb = kRb,
                go = go,
                startPosition = go.transform.position,
                startTop = bounds.max.y,
                height = bounds.max.y - surface,
            });
            bestOwner.colliderToHurdle[col] = hurdle;
        }

        for (int a = 0; a < allAgents.Length; a++)
        {
            var agent = allAgents[a];
            var rows = tempRows[agent];
            var arr = new Hurdle[rows.Count];
            for (int r = 0; r < rows.Count; r++)
            {
                rows[r].row.pieces = rows[r].pieces.ToArray();
                arr[r] = rows[r].row;
            }
            Array.Sort(arr, CompareStartAlong);
            agent.hurdles = arr;

            if (arr.Length == 0 && agent.target != null)
                Debug.LogWarning($"{agent.name}: no '{HurdleTag}' object found in this lane.", agent);
        }
    }

    static float Extent(Bounds bounds, Vector3 direction) =>
        Mathf.Abs(bounds.extents.x * direction.x) + Mathf.Abs(bounds.extents.z * direction.z);

    // Zero-allocation floor raycast using shared static buffer.
    static float FloorHeightBelow(Vector3 point)
    {
        int count = Physics.RaycastNonAlloc(point, Vector3.down, FloorHitsBuffer, 100f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
        float height = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            if (FloorHitsBuffer[i].collider.CompareTag(FloorTag))
                height = Mathf.Max(height, FloorHitsBuffer[i].point.y);
        }
        return height;
    }

    void EnsureRaySensor()
    {
        int rayMask = Physics.DefaultRaycastLayers;
        for (int i = 0; i < AgentLayerNames.Length; i++)
        {
            int layer = LayerMask.NameToLayer(AgentLayerNames[i]);
            if (layer >= 0) rayMask &= ~(1 << layer);
        }

        Transform existing = transform.Find(RaySensorObjectName);
        if (existing != null)
        {
            raySensorTransform = existing;
        }
        else
        {
            var eyes = new GameObject(RaySensorObjectName);
            raySensorTransform = eyes.transform;
            raySensorTransform.SetParent(transform, false);
        }

        Vector3 parentScale = transform.lossyScale;
        raySensorTransform.localScale = new Vector3(
            Mathf.Abs(parentScale.x) > 1e-4f ? 1f / parentScale.x : 1f,
            Mathf.Abs(parentScale.y) > 1e-4f ? 1f / parentScale.y : 1f,
            Mathf.Abs(parentScale.z) > 1e-4f ? 1f / parentScale.z : 1f);
        UpdateRaySensorPose();

        if (!raySensorTransform.TryGetComponent(out RayPerceptionSensorComponent3D raySensor))
            raySensor = raySensorTransform.gameObject.AddComponent<RayPerceptionSensorComponent3D>();

        raySensor.SensorName = "HurdleRays";
        raySensor.DetectableTags = new List<string> { HurdleTag, FloorTag, FinishTag };
        raySensor.RaysPerDirection = raysPerDirection;
        raySensor.MaxRayDegrees = maxRayDegrees;
        raySensor.RayLength = rayLength;
        raySensor.SphereCastRadius = raySphereRadius;
        raySensor.StartVerticalOffset = rayStartVerticalOffset;
        raySensor.EndVerticalOffset = rayEndVerticalOffset;
        raySensor.RayLayerMask = rayMask;
        raySensor.AlternatingRayOrder = false;
        raySensor.UseBatchedRaycasts = true;
    }

    void UpdateRaySensorPose()
    {
        if (raySensorTransform == null || root == null) return;
        raySensorTransform.SetPositionAndRotation(root.position, rayLookRotation);
    }

    public override void OnEpisodeBegin()
    {
        RecordEpisodeStats();
        if (!hurdlesCollected) AssignAllLanesOnce();

        episodeEnding = true;
        spawnGraceSteps = SpawnSettleSteps;

        ApplyHurdleLayout();
        ResetBody();

        nextHurdleIndex = 0;
        hurdlesCleared = 0;
        hurdleHits = 0;
        hitHurdleRows.Clear();
        hitForeignHurdles.Clear();
        maxProgress = Mathf.Max(0f, Progress);
        stallAnchorProgress = maxProgress;
        stallAnchorStep = 0;
        finished = false;
        stalled = false;
        episodeSteps = 0;
        hasPreviousActions = false;

        for (int i = 0; i < feet.Length; i++) feet[i].ResetContacts();
        UpdateRaySensorPose();
    }

    void ApplyHurdleLayout()
    {
        float heightScale = Mathf.Clamp01(Academy.Instance.EnvironmentParameters.GetWithDefault(HurdleHeightParameter, DefaultHurdleHeight));
        bool active = heightScale > MinHurdleHeightScale;

        for (int h = 0; h < hurdles.Length; h++)
        {
            var hurdle = hurdles[h];
            hurdle.active = active;
            float shift = UnityEngine.Random.Range(-hurdleSpacingJitter, hurdleSpacingJitter);
            hurdle.along = hurdle.startAlong + shift;
            hurdle.top = float.NegativeInfinity;

            var pieces = hurdle.pieces;
            for (int p = 0; p < pieces.Length; p++)
            {
                ref var piece = ref pieces[p];
                if (piece.go.activeSelf != active) piece.go.SetActive(active);
                if (!active) continue;

                float sink = (1f - heightScale) * piece.height;
                Vector3 targetPos = piece.startPosition + trackDirection * shift + Vector3.down * sink;
                piece.body.position = targetPos;
                if (piece.kinematicRb != null) piece.kinematicRb.position = targetPos;
                hurdle.top = Mathf.Max(hurdle.top, piece.startTop - sink);
            }
        }
        Array.Sort(hurdles, CompareAlong);
    }

    void ResetBody()
    {
        Quaternion spin = Quaternion.AngleAxis(UnityEngine.Random.Range(-startYawJitter, startYawJitter), Vector3.up);
        Vector3 shift = trackRight * UnityEngine.Random.Range(-startLateralJitter, startLateralJitter);

        for (int i = 0; i < bodyParts.Length; i++)
        {
            ref var part = ref bodyParts[i];
            Vector3 position = rootStartPosition + spin * (part.startPosition - rootStartPosition) + shift;
            Quaternion rotation = spin * part.startRotation;
            part.tr.SetPositionAndRotation(position, rotation);
            part.rb.position = position;
            part.rb.rotation = rotation;
            part.rb.linearVelocity = Vector3.zero;
            part.rb.angularVelocity = Vector3.zero;
            part.rb.WakeUp();
        }

        Quaternion neutralRot = NeutralJointRotation;
        for (int m = 0; m < muscles.Length; m++)
        {
            ref var muscle = ref muscles[m];
            muscle.joint.targetRotation = neutralRot;
            muscle.lastX = float.NaN;
            muscle.lastY = float.NaN;
            muscle.lastZ = float.NaN;
            ApplyDriveStrength(ref muscle, 1f);
        }

        Physics.SyncTransforms();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!bodyCached) CacheBodyParts();
        float safeTrackLength = Mathf.Max(0.01f, trackLength);
        float safeMaxDrift = Mathf.Max(0.1f, maxLaneDrift);
        float invHeightScale = 1f / Mathf.Max(0.01f, heightObservationScale);
        float referenceY = hasTrackFloor ? trackFloorY : rootStartPosition.y;

        Vector3 rootAngVel = rootBody.angularVelocity;
        sensor.AddObservation(toTrackFrame * rootBody.linearVelocity);
        sensor.AddObservation(toTrackFrame * rootAngVel);
        sensor.AddObservation(toTrackFrame * root.forward);
        sensor.AddObservation(toTrackFrame * root.up);
        sensor.AddObservation((root.position.y - referenceY) * invHeightScale);

        float progress = Progress;
        sensor.AddObservation(Mathf.Clamp01(progress / safeTrackLength));
        sensor.AddObservation(Mathf.Clamp(LateralOffset / safeMaxDrift, -1f, 1f));

        var next = NextObservedHurdle(progress);
        if (next != null)
        {
            sensor.AddObservation(Mathf.Clamp((next.along - progress) / hurdleObservationRange, -0.2f, 1f));
            sensor.AddObservation((next.top - referenceY) * invHeightScale);
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(0f);
        }

        for (int m = 0; m < muscles.Length; m++)
        {
            ref var muscle = ref muscles[m];
            sensor.AddObservation(muscle.tr.localRotation);
            sensor.AddObservation(toTrackFrame * (muscle.rb.angularVelocity - rootAngVel));
            sensor.AddObservation(muscle.strength);
        }

        for (int f = 0; f < feet.Length; f++)
        {
            sensor.AddObservation(feet[f].IsTouchingGround ? 1f : 0f);
        }
    }

    Hurdle NextObservedHurdle(float progress)
    {
        int startIdx = nextHurdleIndex > 0 ? nextHurdleIndex - 1 : 0;
        for (int i = startIdx; i < hurdles.Length; i++)
        {
            var h = hurdles[i];
            if (h.active && progress <= h.along + h.halfDepth + TrailingHurdleClearance)
                return h;
        }
        return null;
    }

    bool IsPolicyDriven =>
        behaviorParams != null
        && behaviorParams.BehaviorType != BehaviorType.HeuristicOnly
        && (Academy.Instance.IsCommunicatorOn || behaviorParams.Model != null);

    public override void OnActionReceived(ActionBuffers actions)
    {
        var act = actions.ContinuousActions;
        int i = 0;
        float effort = 0f;
        float actionDeltaSq = 0f;
        bool computeDelta = hasPreviousActions;

        for (int m = 0; m < muscles.Length; m++)
        {
            ref var muscle = ref muscles[m];
            float x = muscle.x ? ReadAction(act, i++, computeDelta, ref actionDeltaSq) : 0f;
            float y = muscle.y ? ReadAction(act, i++, computeDelta, ref actionDeltaSq) : 0f;
            float z = muscle.z ? ReadAction(act, i++, computeDelta, ref actionDeltaSq) : 0f;
            float sAct = ReadAction(act, i++, computeDelta, ref actionDeltaSq);
            effort += x * x + y * y + z * z + sAct * sAct;

            // Dirty-check: skip redundant native PhysX ConfigurableJoint writes on repeated DecisionRequester steps.
            if (x != muscle.lastX || y != muscle.lastY || z != muscle.lastZ)
            {
                muscle.lastX = x;
                muscle.lastY = y;
                muscle.lastZ = z;
                muscle.joint.targetRotation = Quaternion.Euler(
                    muscle.x ? Mathf.Lerp(muscle.lowX, muscle.highX, (x + 1f) * 0.5f) : 0f,
                    y * muscle.limY,
                    z * muscle.limZ);
            }

            float normStrength = (sAct + 1f) * 0.5f;
            if (normStrength != muscle.lastStrength)
            {
                ApplyDriveStrength(ref muscle, normStrength);
            }
        }
        hasPreviousActions = true;
        episodeSteps = StepCount;

        float safeMaxDrift = Mathf.Max(0.1f, maxLaneDrift);
        float absLateral = Mathf.Abs(LateralOffset);
        float driftFraction = Mathf.Clamp01(absLateral / safeMaxDrift);
        float laneAlignment = 1f - 0.5f * driftFraction;
        float lookAt = LookAtTrackReward();

        // 1. Strict forward progress reward (cannot be farmed by going slowly or oscillating).
        float progress = Progress;
        float clampedProgress = Mathf.Clamp(progress, 0f, trackLength);
        float deltaProgress = clampedProgress - maxProgress;
        if (deltaProgress > 0f)
        {
            AddReward(progressRewardScale * (deltaProgress / trackLength) * lookAt * laneAlignment);
            maxProgress = clampedProgress;
        }

        if (maxProgress >= stallAnchorProgress + minProgressAdvance)
        {
            stallAnchorProgress = maxProgress;
            stallAnchorStep = StepCount;
        }

        // 2. Mass-weighted CoM forward sprint shaping.
        AddReward(velocityShapingScale * ForwardSpeedReward() * lookAt * laneAlignment);

        // 3. Lane centering, time efficiency, effort, and smoothness costs.
        AddReward(-laneCenterPenalty * driftFraction * driftFraction - timePenalty);
        if (actionCount > 0)
        {
            float invAct = 1f / actionCount;
            AddReward((-effortPenalty * effort - smoothnessPenalty * actionDeltaSq) * invAct);
        }

        UpdateClearedHurdles(progress);

        float fallThresholdY = hasTrackFloor
            ? Mathf.Min(rootStartPosition.y - FallMargin, trackFloorY + MinHeightAboveFloor)
            : rootStartPosition.y - FallMargin;

        bool hasStalled = maxStallSteps > 0 && IsPolicyDriven && (StepCount - stallAnchorStep >= maxStallSteps);
        if (spawnGraceSteps <= 0 && (root.position.y < fallThresholdY || !IsUpright || absLateral > safeMaxDrift || hasStalled))
        {
            stalled = hasStalled;
            EndEpisodeWithFall();
            return;
        }

        if (progress >= trackLength) OnFinishReached();
    }

    float ReadAction(ActionSegment<float> act, int index, bool computeDelta, ref float deltaSqSum)
    {
        float clamped = Mathf.Clamp(act[index], -1f, 1f);
        if (previousActions != null && index < previousActions.Length)
        {
            if (computeDelta)
            {
                float delta = clamped - previousActions[index];
                deltaSqSum += delta * delta;
            }
            previousActions[index] = clamped;
        }
        return clamped;
    }

    // Mass-weighted Center-of-Mass (CoM) velocity: cannot be gamed by swinging light hands/feet.
    float ForwardSpeedReward()
    {
        Vector3 comVelocity = Vector3.zero;
        for (int i = 0; i < bodyParts.Length; i++)
        {
            comVelocity += bodyParts[i].rb.linearVelocity * bodyParts[i].mass;
        }
        comVelocity *= invTotalMass;

        float forwardSpeed = Vector3.Dot(comVelocity, trackDirection);
        if (forwardSpeed <= 0f) return 0f;

        float lateralSpeed = Vector3.Dot(comVelocity, trackRight);
        float speedScore = Mathf.Clamp01(forwardSpeed / Mathf.Max(0.1f, sprintTargetSpeed));
        float lateralDamp = Mathf.Exp(-0.25f * lateralSpeed * lateralSpeed);
        return speedScore * lateralDamp;
    }

    float LookAtTrackReward()
    {
        Vector3 fwd = root.forward;
        Vector3 heading = new Vector3(fwd.x, 0f, fwd.z);
        float sqrMag = heading.sqrMagnitude;
        if (sqrMag < 1e-6f) return 0f;
        return (Vector3.Dot(heading * (1f / Mathf.Sqrt(sqrMag)), trackDirection) + 1f) * 0.5f;
    }

    void UpdateClearedHurdles(float progress)
    {
        while (nextHurdleIndex < hurdles.Length && progress > hurdles[nextHurdleIndex].along + hurdles[nextHurdleIndex].halfDepth)
        {
            if (hurdles[nextHurdleIndex].active)
            {
                hurdlesCleared++;
                AddReward(hurdleClearedReward);
            }
            nextHurdleIndex++;
        }
    }

    void ApplyDriveStrength(ref Muscle muscle, float strength)
    {
        muscle.strength = strength;
        muscle.lastStrength = strength;
        muscle.joint.slerpDrive = new JointDrive
        {
            positionSpring = muscleStrength,
            positionDamper = muscleDamper,
            maximumForce = strength * maxMuscleForce,
            useAcceleration = true,
        };
    }

    protected virtual void FixedUpdate()
    {
        if (decisionRequester == null || decisionRequester.Agent != this) BindDecisionRequester();
        if (spawnGraceSteps > 0) spawnGraceSteps--;
        episodeEnding = false;
        UpdateRaySensorPose();
    }

    // --- Contacts reported by BodyPartContact ---

    public void OnHurdleHit(Collider hurdleCollider)
    {
        if (episodeEnding) return;

        if (colliderToHurdle.TryGetValue(hurdleCollider, out var hurdleRow))
        {
            if (!hitHurdleRows.Add(hurdleRow)) return;
        }
        else if (!hitForeignHurdles.Add(hurdleCollider))
        {
            return;
        }

        hurdleHits++;
        AddReward(hurdleHitPenalty);
    }

    public void OnBodyPartTouchedFloor()
    {
        if (episodeEnding || spawnGraceSteps > 0 || !ShouldEndOnFloorContact()) return;
        EndEpisodeWithFall();
    }

    public void OnFinishReached()
    {
        if (episodeEnding) return;
        episodeEnding = true;
        finished = true;

        float remainingFraction = MaxStep > 0 ? Mathf.Clamp01(1f - (float)episodeSteps / MaxStep) : 0f;
        AddReward(targetReachedReward + finishSpeedBonus * remainingFraction);
        EndEpisode();
    }

    void EndEpisodeWithFall()
    {
        episodeEnding = true;
        AddReward(fallPenalty);
        EndEpisode();
    }

    void RecordEpisodeStats()
    {
        if (episodeSteps == 0 && StepCount == 0) return;
        if (statHurdlesCleared == null) SetBehaviorName(behaviorName ?? "Agent");
        int steps = episodeSteps > 0 ? episodeSteps : StepCount;

        var stats = Academy.Instance.StatsRecorder;
        stats.Add(statHurdlesCleared, hurdlesCleared);
        stats.Add(statHurdleHits, hurdleHits);
        stats.Add(statTrackProgress, finished ? 1f : Mathf.Clamp01(maxProgress / trackLength));
        stats.Add(statFinishRate, finished ? 1f : 0f);
        stats.Add(statStallRate, stalled ? 1f : 0f);
        if (finished) stats.Add(statFinishTime, steps * Time.fixedDeltaTime);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (!bodyCached) CacheBodyParts();

        var act = actionsOut.ContinuousActions;
        for (int k = 0; k < act.Length; k++) act[k] = 0f;
        if (muscles.Length == 0 || act.Length == 0) return;

        var keyboard = Keyboard.current;
        float driveX = 0f;
        float driveY = 0f;
        bool flexAll = false;

        if (keyboard != null)
        {
            bool previousPressed = keyboard.leftBracketKey.isPressed;
            bool nextPressed = keyboard.rightBracketKey.isPressed;
            if ((previousPressed && !previousKeyState) || (nextPressed && !nextKeyState))
            {
                int step = nextPressed ? 1 : -1;
                heuristicMuscle = (heuristicMuscle + step + muscles.Length) % muscles.Length;
                Debug.Log($"{name}: heuristic joint = {muscles[heuristicMuscle].joint.name}", this);
            }
            previousKeyState = previousPressed;
            nextKeyState = nextPressed;

            driveX = (keyboard.upArrowKey.isPressed || keyboard.wKey.isPressed ? 1f : 0f)
                   - (keyboard.downArrowKey.isPressed || keyboard.sKey.isPressed ? 1f : 0f);
            driveY = (keyboard.rightArrowKey.isPressed || keyboard.dKey.isPressed ? 1f : 0f)
                   - (keyboard.leftArrowKey.isPressed || keyboard.aKey.isPressed ? 1f : 0f);
            flexAll = keyboard.spaceKey.isPressed;
        }

        int i = 0;
        for (int m = 0; m < muscles.Length; m++)
        {
            ref var muscle = ref muscles[m];
            bool selected = m == heuristicMuscle;
            if (muscle.x) act[i++] = flexAll ? Mathf.Sin(Time.time * 6f + m * 0.8f) : (selected ? driveX : 0f);
            if (muscle.y) act[i++] = selected ? driveY : 0f;
            if (muscle.z) i++;
            act[i++] = 1f; // full strength
        }
    }
}
