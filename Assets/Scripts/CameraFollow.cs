using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;

// Frame-rate independent broadcast follow camera.
//   Tab : cycle through agents
//   L   : toggle auto-follow race leader
//   H   : toggle live broadcast telemetry HUD
public class CameraFollow : MonoBehaviour
{
    [SerializeField] Transform target;
    [Tooltip("Camera position relative to the target.")]
    [SerializeField] Vector3 offset = new Vector3(0f, 5f, -10f);
    [Tooltip("Higher = snappier position follow (1/s).")]
    [FormerlySerializedAs("smoothSpeed")]
    [SerializeField] float followSharpness = 6f;
    [Tooltip("Higher = snappier rotation follow (1/s) — damps ragdoll vertical bobbing.")]
    [SerializeField] float rotationSharpness = 8f;
    [SerializeField] bool followLeader;
    [SerializeField] bool showBroadcastHud = true;

    HurdleAgentBase[] agents;
    HurdleAgentBase activeAgent;
    int agentIndex = -1;
    float smoothedLookY;
    GUIStyle hudBoxStyle;
    GUIStyle hudTitleStyle;
    GUIStyle hudBodyStyle;

    void Awake()
    {
        if (!Application.isBatchMode) return;
        if (TryGetComponent(out Camera cam)) cam.enabled = false;
        if (TryGetComponent(out AudioListener listener)) listener.enabled = false;
        enabled = false;
    }

    void Start()
    {
        agents = FindObjectsByType<HurdleAgentBase>(FindObjectsSortMode.InstanceID);
        if (agents.Length > 0)
        {
            if (target == null)
            {
                SelectAgent(0);
            }
            else
            {
                for (int i = 0; i < agents.Length; i++)
                {
                    if (agents[i].Root == target || agents[i].transform == target)
                    {
                        activeAgent = agents[i];
                        agentIndex = i;
                        break;
                    }
                }
            }
        }
        if (target != null) smoothedLookY = target.position.y;
    }

    void SelectAgent(int index)
    {
        if (agents == null || agents.Length == 0) return;
        agentIndex = (index + agents.Length) % agents.Length;
        activeAgent = agents[agentIndex];
        target = activeAgent.Root != null ? activeAgent.Root : activeAgent.transform;
        if (target != null) smoothedLookY = target.position.y;
    }

    void LateUpdate()
    {
        var keyboard = Keyboard.current;
        if (keyboard != null && agents != null && agents.Length > 0)
        {
            if (keyboard.tabKey.wasPressedThisFrame)
            {
                followLeader = false;
                SelectAgent(agentIndex + 1);
            }
            if (keyboard.lKey.wasPressedThisFrame) followLeader = !followLeader;
            if (keyboard.hKey.wasPressedThisFrame) showBroadcastHud = !showBroadcastHud;
        }

        if (followLeader && agents != null && agents.Length > 0)
        {
            int bestIdx = agentIndex >= 0 ? agentIndex : 0;
            float bestProgress = -1f;
            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] != null && agents[i].TrackProgress01 > bestProgress)
                {
                    bestProgress = agents[i].TrackProgress01;
                    bestIdx = i;
                }
            }
            if (bestIdx != agentIndex) SelectAgent(bestIdx);
        }

        if (target == null) return;

        float dt = Time.deltaTime;
        float posT = 1f - Mathf.Exp(-followSharpness * dt);
        float rotT = 1f - Mathf.Exp(-rotationSharpness * dt);

        smoothedLookY = Mathf.Lerp(smoothedLookY, target.position.y, posT * 0.5f);
        Vector3 smoothedAnchor = new Vector3(target.position.x, smoothedLookY, target.position.z);

        transform.position = Vector3.Lerp(transform.position, smoothedAnchor + offset, posT);
        Vector3 lookDir = smoothedAnchor - transform.position;
        if (lookDir.sqrMagnitude > 1e-4f)
        {
            Quaternion desiredRot = Quaternion.LookRotation(lookDir, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, desiredRot, rotT);
        }
    }

    void OnGUI()
    {
        if (!showBroadcastHud || activeAgent == null) return;

        if (hudBoxStyle == null)
        {
            hudBoxStyle = new GUIStyle(GUI.skin.box) { padding = new RectOffset(14, 14, 10, 10) };
            hudTitleStyle = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold };
            hudTitleStyle.normal.textColor = Color.white;
            hudBodyStyle = new GUIStyle(GUI.skin.label) { fontSize = 13 };
            hudBodyStyle.normal.textColor = new Color(0.9f, 0.94f, 1f);
        }

        float speedMs = activeAgent.CurrentSpeed;
        float speedKmh = speedMs * 3.6f;
        string modeTag = followLeader ? "LEADER" : $"CAM {agentIndex + 1}/{(agents != null ? agents.Length : 1)}";

        GUILayout.BeginArea(new Rect(16f, 16f, 300f, 130f), hudBoxStyle);
        GUILayout.Label($"[{modeTag}] {activeAgent.name} ({activeAgent.BehaviorName})", hudTitleStyle);
        GUILayout.Label($"Chrono : {activeAgent.EpisodeTimeSeconds:00.00} s   |   Speed : {speedMs:0.0} m/s ({speedKmh:0.0} km/h)", hudBodyStyle);
        GUILayout.Label($"Hurdles : {activeAgent.HurdlesClearedCount}        |   Track : {activeAgent.TrackProgress01 * 100f:0.0}%", hudBodyStyle);
        GUILayout.Label("[Tab] Next Agent   [L] Auto-Leader   [H] Hide HUD", hudBodyStyle);
        GUILayout.EndArea();
    }
}
