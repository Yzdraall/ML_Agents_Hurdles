using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;

// Frame-rate independent follow camera. Press Tab to cycle through the agents of the scene.
public class CameraFollow : MonoBehaviour
{
    [SerializeField] Transform target;
    [Tooltip("Camera position relative to the target.")]
    [SerializeField] Vector3 offset = new Vector3(0f, 5f, -10f);
    [Tooltip("Higher = snappier follow (1/s).")]
    [FormerlySerializedAs("smoothSpeed")]
    [SerializeField] float followSharpness = 5f;

    HurdleAgentBase[] agents;
    int agentIndex = -1;

    void Start()
{
    agents = FindObjectsByType<HurdleAgentBase>(FindObjectsSortMode.InstanceID);
    if (target == null && agents.Length > 0)
    {
        agentIndex = 0;
        target = agents[0].Root;
    }
}

    void LateUpdate()
    {
        var keyboard = Keyboard.current;
        if (keyboard != null && keyboard.tabKey.wasPressedThisFrame && agents.Length > 0)
        {
            agentIndex = (agentIndex + 1) % agents.Length;
            target = agents[agentIndex].Root;
        }

        if (target == null) return;

        float t = 1f - Mathf.Exp(-followSharpness * Time.deltaTime);
        transform.position = Vector3.Lerp(transform.position, target.position + offset, t);
        transform.LookAt(target);
    }
}
