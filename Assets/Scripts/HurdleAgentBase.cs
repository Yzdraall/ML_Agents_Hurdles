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
// body-part caching and full pose reset, joint actuation (multi-axis targets + strength per joint),
// track-relative observations, ray perception, rewards, curriculum, randomisation and custom stats.
[DefaultExecutionOrder(-100)]
public abstract class HurdleAgentBase : Agent
{
    public const string HurdleTag = "hurdle";
    public const string FloorTag = "floor";
    public const string FinishTag = "finish";

    // Physics layers holding agent bodies: excluded from the ray perception.
    static readonly string[] AgentLayerNames = { "Walker", "Crawler" };

    // Root velocity (3) + root angular velocity (3) + root forward (3) + root up (3)
    // + root height (1) + track progress (1) + next hurdle distance & height (2).
    const int CommonObservationCount = 16;
    // Local rotation (4) + angular velocity relative to the root (3) + current strength (1).
    const int ObservationsPerMuscle = 8;
    const float MinHurdleHeightScale = 0.01f;
    const int SpawnSettleSteps = 30;

    static readonly FieldInfo RequesterAgentField =
        typeof(DecisionRequester).GetField("m_Agent", BindingFlags.Instance | BindingFlags.NonPublic);

    [Header("Track")]
    [Tooltip("Finish line of this agent's lane. The track runs straight to the nearest point of its collider.")]
    [SerializeField] Transform target;
    [Tooltip("Root body part (hips / body). Auto-detected (first rigidbody) if empty.")]
    [FormerlySerializedAs("hips")]
    [SerializeField] Transform root;
    [Tooltip("A tagged hurdle belongs to this lane when its span comes within this lateral distance (m) of the lane centre line.")]
    [SerializeField] float laneMargin = 2f;
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
    [Tooltip("Speed (m/s) along the track that maximises the speed reward.")]
    [SerializeField] float targetSpeed = 5f;
    [Tooltip("Weight of the per-step reward speedMatch x lookAtTrack (both in [0, 1]).")]
    [SerializeField] float speedRewardScale = 0.01f;
    [Tooltip("Weight of the effort cost: mean of the squared actions.")]
    [SerializeField] float effortPenalty = 0.002f;
    [SerializeField] float hurdleClearedReward = 0.5f;
    [SerializeField] float targetReachedReward = 10f;
    [SerializeField] float fallPenalty = -1f;
    [SerializeField] float hurdleHitPenalty = -0.5f;

    [Header("Termination")]
    [Tooltip("Episode ends when dot(root.up, world up) drops below this value.")]
    [SerializeField, Range(-1f, 1f)] float uprightThreshold = 0.15f;

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
        public Vector3 startPosition;
        public Quaternion startRotation;
    }

    class Muscle
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public bool x, y, z;
        public float strength;
        public int ActionCount => (x ? 1 : 0) + (y ? 1 : 0) + (z ? 1 : 0) + 1;
    }

    // A hurdle of the lane: the tagged objects standing at the same distance along the track.
    class Hurdle
    {
        public readonly List<HurdlePiece> pieces = new List<HurdlePiece>();
        public float startAlong;
        public float halfDepth;
        public float along;
        public float top = float.NegativeInfinity;
        public bool active = true;
    }

    struct HurdlePiece
    {
        public Transform body;
        public Vector3 startPosition;
        public float startTop;
        // Height of the top above the running surface: the part scaled by the curriculum.
        public float height;
    }

    static readonly HashSet<Collider> claimedHurdles = new HashSet<Collider>();

    readonly List<BodyPart> bodyParts = new List<BodyPart>();
    readonly List<Muscle> muscles = new List<Muscle>();
    readonly List<GroundContact> feet = new List<GroundContact>();
    readonly List<Hurdle> hurdles = new List<Hurdle>();

    Rigidbody rootBody;
    Vector3 rootStartPosition;
    Vector3 rootLocalUp;
    float trackFloorY;
    bool hasTrackFloor;
    Vector3 trackDirection;
    Vector3 trackRight;
    Quaternion toTrackFrame;
    float trackLength;
    int actionCount;
    string behaviorName;
    Transform raySensorTransform;
    bool trackReady;
    DecisionRequester decisionRequester;

    // Episode state.
    readonly HashSet<Collider> hitHurdles = new HashSet<Collider>();
    bool episodeEnding;
    int spawnGraceSteps;
    int nextHurdleIndex;
    int hurdlesCleared;
    int hurdleHits;
    float maxProgress;
    bool finished;
    int episodeSteps;

    // Heuristic state.
    int heuristicMuscle;
    bool previousKeyState;
    bool nextKeyState;

    public Transform Root => root;

    protected abstract float FallMargin { get; }
    protected abstract string BodyLayerName { get; }
    // Environment parameter (config/hurdles.yaml) scaling the hurdle height: 0 = flat track, 1 = real height.
    protected abstract string HurdleHeightParameter { get; }
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
            foreach (var foot in feet)
            {
                if (foot.IsTouchingGround) return true;
            }
            return false;
        }
    }

    float Progress => Vector3.Dot(root.position - rootStartPosition, trackDirection);

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ClearStaticState() => claimedHurdles.Clear();

    protected override void OnEnable()
    {
        RemoveStrayBaseAgents();
        EnsureDecisionRequester();

        // Body parts must be known before the Agent creates its policy and sensors.
        CacheBodyParts();
        SyncBrainParameters();
        base.OnEnable();
        BindDecisionRequester();
    }

    void Start() => BindDecisionRequester();

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
        if (trackReady) return;
        CacheBodyParts();
        if (MaxStep == 0) MaxStep = 5250;

        trackReady = true;
        if (target == null)
        {
            Debug.LogError($"{name}: 'target' is not assigned in the Inspector. Using root forward as track direction.", this);
            Vector3 forward = Vector3.ProjectOnPlane(root.forward, Vector3.up);
            trackDirection = forward.sqrMagnitude > 1e-4f ? forward.normalized : Vector3.forward;
            trackRight = Vector3.Cross(Vector3.up, trackDirection);
            toTrackFrame = Quaternion.Inverse(Quaternion.LookRotation(trackDirection, Vector3.up));
            trackLength = 110f;
        }
        else
        {
            Vector3 toFinish = Vector3.ProjectOnPlane(ClosestFinishPoint() - rootStartPosition, Vector3.up);
            trackDirection = toFinish.sqrMagnitude > 1e-4f ? toFinish.normalized : Vector3.forward;
            trackRight = Vector3.Cross(Vector3.up, trackDirection);
            toTrackFrame = Quaternion.Inverse(Quaternion.LookRotation(trackDirection, Vector3.up));
            trackLength = Mathf.Max(0.01f, toFinish.magnitude);
            CollectLaneHurdles();
        }

        CreateRaySensor();
    }

    Vector3 ClosestFinishPoint()
    {
        if (!target.TryGetComponent(out Collider finish)) return target.position;
        // Collider.ClosestPoint does not support concave mesh colliders.
        return finish is MeshCollider { convex: false }
            ? finish.bounds.ClosestPoint(rootStartPosition)
            : finish.ClosestPoint(rootStartPosition);
    }

    void CacheBodyParts()
    {
        if (bodyParts.Count > 0) return;

        // Auto-detect root body part: search for main body names before defaulting to first Rigidbody.
        if (root == null)
        {
            foreach (var rb in GetComponentsInChildren<Rigidbody>())
            {
                string n = rb.name.ToLowerInvariant();
                if (n == "body" || n == "hips" || n == "torso" || n == "pelvis")
                {
                    root = rb.transform;
                    break;
                }
            }
            if (root == null) root = GetComponentInChildren<Rigidbody>()?.transform;
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

        // Parents are returned before their children, so restoring world poses in this order is safe.
        foreach (var rb in GetComponentsInChildren<Rigidbody>())
        {
            bodyParts.Add(new BodyPart { rb = rb, startPosition = rb.transform.position, startRotation = rb.transform.rotation });

            bool isFoot = IsFootPart(rb);
            if (isFoot && !rb.TryGetComponent(out GroundContact _)) rb.gameObject.AddComponent<GroundContact>();

            if (!rb.TryGetComponent(out BodyPartContact contact))
            {
                contact = rb.gameObject.AddComponent<BodyPartContact>();
            }
            contact.CanTouchFloor = CanTouchFloor(rb);
            contact.Agent = this;
        }

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
                for (int i = 0; i < bodyParts.Count; i++)
                {
                    var part = bodyParts[i];
                    part.startPosition += lift;
                    bodyParts[i] = part;
                }
            }
        }

        GetComponentsInChildren(feet);

        actionCount = 0;
        foreach (var joint in GetComponentsInChildren<ConfigurableJoint>())
        {
            var muscle = new Muscle
            {
                joint = joint,
                rb = joint.GetComponent<Rigidbody>(),
                x = joint.angularXMotion != ConfigurableJointMotion.Locked,
                y = joint.angularYMotion != ConfigurableJointMotion.Locked,
                z = joint.angularZMotion != ConfigurableJointMotion.Locked,
            };
            if (!muscle.x && !muscle.y && !muscle.z) continue;
            // Every muscle uses the same drive model (see SetStrength), whatever the prefab or scene settings.
            joint.rotationDriveMode = RotationDriveMode.Slerp;
            muscles.Add(muscle);
            actionCount += muscle.ActionCount;
            SetStrength(muscle, 1f);
        }
    }

    // Keeps the Behavior Parameters consistent with the body: observation and action sizes depend on
    // the number of actuated joints, their free axes and the number of feet.
    void SyncBrainParameters()
    {
        var behavior = GetComponent<BehaviorParameters>();
        behaviorName = behavior.BehaviorName;

        var brain = behavior.BrainParameters;
        int observationCount = CommonObservationCount + muscles.Count * ObservationsPerMuscle + feet.Count;
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

    void CollectLaneHurdles()
    {
        foreach (var go in GameObject.FindGameObjectsWithTag(HurdleTag))
        {
            var col = go.GetComponent<Collider>();
            if (col == null) col = go.GetComponentInChildren<Collider>();
            if (col == null) continue;

            Bounds bounds = col.bounds;
            Vector3 offset = bounds.center - rootStartPosition;
            float along = Vector3.Dot(offset, trackDirection);
            if (along <= 0f || along >= trackLength) continue;

            // The agent may hit any object whose span comes within laneMargin of the lane centre line.
            if (Mathf.Abs(Vector3.Dot(offset, trackRight)) - Extent(bounds, trackRight) > laneMargin) continue;

            // A hurdle is driven (moved / sunk) by a single agent.
            if (!claimedHurdles.Add(col)) continue;

            // Objects overlapping along the track form a single hurdle (one row of the track).
            float halfDepth = Extent(bounds, trackDirection);
            float near = along - halfDepth;
            float far = along + halfDepth;
            var hurdle = hurdles.Find(h => near < h.startAlong + h.halfDepth && far > h.startAlong - h.halfDepth);
            if (hurdle == null)
            {
                hurdle = new Hurdle();
                hurdles.Add(hurdle);
            }
            else
            {
                near = Mathf.Min(near, hurdle.startAlong - hurdle.halfDepth);
                far = Mathf.Max(far, hurdle.startAlong + hurdle.halfDepth);
            }
            hurdle.startAlong = hurdle.along = (near + far) * 0.5f;
            hurdle.halfDepth = (far - near) * 0.5f;
            hurdle.top = Mathf.Max(hurdle.top, bounds.max.y);

            // The curriculum scales the height of the top above the running surface (above the box bottom if no floor is found).
            float floor = FloorHeightBelow(new Vector3(bounds.center.x, bounds.max.y, bounds.center.z));
            float surface = float.IsNegativeInfinity(floor) ? bounds.min.y : floor;
            hurdle.pieces.Add(new HurdlePiece
            {
                body = go.transform,
                startPosition = go.transform.position,
                startTop = bounds.max.y,
                height = bounds.max.y - surface,
            });
        }
        hurdles.Sort((a, b) => a.startAlong.CompareTo(b.startAlong));

        if (hurdles.Count == 0) Debug.LogWarning($"{name}: no '{HurdleTag}' object found in this lane.", this);
    }

    // Half size of a bounding box along a horizontal direction.
    static float Extent(Bounds bounds, Vector3 direction) =>
        Mathf.Abs(bounds.extents.x * direction.x) + Mathf.Abs(bounds.extents.z * direction.z);

    // Height of the highest floor-tagged surface below a point (negative infinity if there is none).
    static float FloorHeightBelow(Vector3 point)
    {
        float height = float.NegativeInfinity;
        foreach (var hit in Physics.RaycastAll(point, Vector3.down, Mathf.Infinity, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
        {
            if (hit.collider.CompareTag(FloorTag)) height = Mathf.Max(height, hit.point.y);
        }
        return height;
    }

    void CreateRaySensor()
    {
        int rayMask = Physics.DefaultRaycastLayers;
        foreach (var layerName in AgentLayerNames)
        {
            int layer = LayerMask.NameToLayer(layerName);
            if (layer >= 0) rayMask &= ~(1 << layer);
        }

        // The sensor lives on its own child so it can be kept level and aligned with the track.
        var eyes = new GameObject("HurdleRaySensor");
        raySensorTransform = eyes.transform;
        raySensorTransform.SetParent(transform, false);
        Vector3 parentScale = transform.lossyScale;
        raySensorTransform.localScale = new Vector3(1f / parentScale.x, 1f / parentScale.y, 1f / parentScale.z);
        UpdateRaySensorPose();

        var raySensor = eyes.AddComponent<RayPerceptionSensorComponent3D>();
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
    }

    void UpdateRaySensorPose()
    {
        if (raySensorTransform == null) return;
        raySensorTransform.SetPositionAndRotation(root.position, Quaternion.LookRotation(trackDirection, Vector3.up));
    }

    public override void OnEpisodeBegin()
    {
        RecordEpisodeStats();
        episodeEnding = true;
        spawnGraceSteps = SpawnSettleSteps;

        ApplyHurdleLayout();
        ResetBody();

        // episodeEnding is cleared at the next FixedUpdate: contacts still pending in this physics step
        // belong to the previous episode.
        nextHurdleIndex = 0;
        hurdlesCleared = 0;
        hurdleHits = 0;
        hitHurdles.Clear();
        maxProgress = 0f;
        finished = false;
        episodeSteps = 0;

        foreach (var foot in feet) foot.ResetContacts();
        UpdateRaySensorPose();
    }

    void ApplyHurdleLayout()
    {
        float heightScale = Mathf.Clamp01(Academy.Instance.EnvironmentParameters.GetWithDefault(HurdleHeightParameter, 1f));
        bool active = heightScale > MinHurdleHeightScale;

        foreach (var hurdle in hurdles)
        {
            hurdle.active = active;
            float shift = Random.Range(-hurdleSpacingJitter, hurdleSpacingJitter);
            hurdle.along = hurdle.startAlong + shift;
            hurdle.top = float.NegativeInfinity;

            // The pieces move together. Lower hurdles are sunk into the track rather than rescaled,
            // which works whatever the mesh orientation.
            foreach (var piece in hurdle.pieces)
            {
                piece.body.gameObject.SetActive(active);
                float sink = (1f - heightScale) * piece.height;
                piece.body.position = piece.startPosition + trackDirection * shift + Vector3.down * sink;
                hurdle.top = Mathf.Max(hurdle.top, piece.startTop - sink);
            }
        }
        hurdles.Sort((a, b) => a.along.CompareTo(b.along));
    }

    void ResetBody()
    {
        Quaternion spin = Quaternion.AngleAxis(Random.Range(-startYawJitter, startYawJitter), Vector3.up);
        Vector3 shift = trackRight * Random.Range(-startLateralJitter, startLateralJitter);

        foreach (var part in bodyParts)
        {
            Vector3 position = rootStartPosition + spin * (part.startPosition - rootStartPosition) + shift;
            Quaternion rotation = spin * part.startRotation;
            part.rb.transform.SetPositionAndRotation(position, rotation);
            part.rb.position = position;
            part.rb.rotation = rotation;
            part.rb.linearVelocity = Vector3.zero;
            part.rb.angularVelocity = Vector3.zero;
            part.rb.WakeUp();
        }

        foreach (var muscle in muscles)
        {
            muscle.joint.targetRotation = NeutralJointRotation;
            SetStrength(muscle, 1f);
        }

        Physics.SyncTransforms();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (bodyParts.Count == 0) CacheBodyParts();
        float safeTrackLength = Mathf.Max(0.01f, trackLength);

        sensor.AddObservation(toTrackFrame * rootBody.linearVelocity);
        sensor.AddObservation(toTrackFrame * rootBody.angularVelocity);
        sensor.AddObservation(toTrackFrame * root.forward);
        sensor.AddObservation(toTrackFrame * root.up);
        sensor.AddObservation((root.position.y - rootStartPosition.y) / heightObservationScale);

        float progress = Progress;
        sensor.AddObservation(Mathf.Clamp01(progress / safeTrackLength));

        var next = NextHurdle();
        if (next != null)
        {
            sensor.AddObservation(Mathf.Clamp01((next.along - progress) / hurdleObservationRange));
            sensor.AddObservation((next.top - rootStartPosition.y) / heightObservationScale);
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(0f);
        }

        foreach (var muscle in muscles)
        {
            sensor.AddObservation(muscle.joint.transform.localRotation);
            sensor.AddObservation(toTrackFrame * (muscle.rb.angularVelocity - rootBody.angularVelocity));
            sensor.AddObservation(muscle.strength);
        }

        foreach (var foot in feet)
        {
            sensor.AddObservation(foot.IsTouchingGround ? 1f : 0f);
        }
    }

    Hurdle NextHurdle()
    {
        for (int i = nextHurdleIndex; i < hurdles.Count; i++)
        {
            if (hurdles[i].active) return hurdles[i];
        }
        return null;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var act = actions.ContinuousActions;
        int i = 0;
        float effort = 0f;

        foreach (var muscle in muscles)
        {
            var joint = muscle.joint;
            float x = muscle.x ? Mathf.Clamp(act[i++], -1f, 1f) : 0f;
            float y = muscle.y ? Mathf.Clamp(act[i++], -1f, 1f) : 0f;
            float z = muscle.z ? Mathf.Clamp(act[i++], -1f, 1f) : 0f;
            float strength = Mathf.Clamp(act[i++], -1f, 1f);
            effort += x * x + y * y + z * z + strength * strength;

            // Targets stay inside each joint's own limits.
            joint.targetRotation = Quaternion.Euler(
                muscle.x ? Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, (x + 1f) * 0.5f) : 0f,
                y * joint.angularYLimit.limit,
                z * joint.angularZLimit.limit);
            SetStrength(muscle, (strength + 1f) * 0.5f);
        }

        episodeSteps = StepCount;

        // Reward: speed along the track matching the target speed, while facing the track.
        AddReward(speedRewardScale * SpeedMatchReward() * LookAtTrackReward());
        if (actionCount > 0) AddReward(-effortPenalty * effort / actionCount);

        float progress = Progress;
        maxProgress = Mathf.Max(maxProgress, progress);
        UpdateClearedHurdles(progress);

        float fallThresholdY = hasTrackFloor
            ? Mathf.Min(rootStartPosition.y - FallMargin, trackFloorY + MinHeightAboveFloor)
            : rootStartPosition.y - FallMargin;
        if (spawnGraceSteps <= 0 && (root.position.y < fallThresholdY || !IsUpright))
        {
            EndEpisodeWithFall();
            return;
        }

        if (progress >= trackLength) OnFinishReached();
    }

    float SpeedMatchReward()
    {
        Vector3 averageVelocity = Vector3.zero;
        foreach (var part in bodyParts) averageVelocity += part.rb.linearVelocity;
        averageVelocity /= bodyParts.Count;

        float error = Mathf.Clamp(Vector3.Distance(averageVelocity, trackDirection * targetSpeed), 0f, targetSpeed);
        float normalisedError = error / targetSpeed;
        return Mathf.Pow(1f - normalisedError * normalisedError, 2f);
    }

    float LookAtTrackReward()
    {
        Vector3 heading = Vector3.ProjectOnPlane(root.forward, Vector3.up);
        if (heading.sqrMagnitude < 1e-6f) return 0f;
        return (Vector3.Dot(heading.normalized, trackDirection) + 1f) * 0.5f;
    }

    void UpdateClearedHurdles(float progress)
    {
        while (nextHurdleIndex < hurdles.Count && progress > hurdles[nextHurdleIndex].along + hurdles[nextHurdleIndex].halfDepth)
        {
            if (hurdles[nextHurdleIndex].active)
            {
                hurdlesCleared++;
                AddReward(hurdleClearedReward);
            }
            nextHurdleIndex++;
        }
    }

    // The agent owns the drives: slerp drive in acceleration mode, so the same spring and damper
    // act alike on light and heavy body parts.
    void SetStrength(Muscle muscle, float strength)
    {
        muscle.strength = strength;
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
        BindDecisionRequester();
        if (spawnGraceSteps > 0) spawnGraceSteps--;
        // Contacts reported later in this physics step may still concern the previous episode's pose.
        episodeEnding = false;
        UpdateRaySensorPose();
    }

    // --- Contacts reported by BodyPartContact ---

    public void OnHurdleHit(Collider hurdleCollider)
    {
        if (episodeEnding || !hitHurdles.Add(hurdleCollider)) return;
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
        AddReward(targetReachedReward);
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
        int steps = episodeSteps > 0 ? episodeSteps : StepCount;

        var stats = Academy.Instance.StatsRecorder;
        stats.Add($"{behaviorName}/Hurdles Cleared", hurdlesCleared);
        stats.Add($"{behaviorName}/Hurdle Hits", hurdleHits);
        stats.Add($"{behaviorName}/Track Progress", finished ? 1f : Mathf.Clamp01(maxProgress / trackLength));
        stats.Add($"{behaviorName}/Finish Rate", finished ? 1f : 0f);
        if (finished) stats.Add($"{behaviorName}/Finish Time (s)", steps * Time.fixedDeltaTime);
    }

    // Manual control for debugging:
    //   [ / ]        select the previous / next joint (name logged in the console)
    //   Up / Down    (or W / S) drive the selected joint around its X axis
    //   Left / Right (or A / D) drive the selected joint around its Y axis
    //   Space        drive an oscillating wave across all X joints
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (muscles.Count == 0) CacheBodyParts();

        var act = actionsOut.ContinuousActions;
        for (int k = 0; k < act.Length; k++) act[k] = 0f;
        if (muscles.Count == 0 || act.Length == 0) return;

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
                heuristicMuscle = (heuristicMuscle + step + muscles.Count) % muscles.Count;
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
        for (int m = 0; m < muscles.Count; m++)
        {
            var muscle = muscles[m];
            bool selected = m == heuristicMuscle;
            if (muscle.x) act[i++] = flexAll ? Mathf.Sin(Time.time * 6f + m * 0.8f) : (selected ? driveX : 0f);
            if (muscle.y) act[i++] = selected ? driveY : 0f;
            if (muscle.z) i++;
            act[i++] = 1f; // full strength
        }
    }
}
