using UnityEngine;
using UnityEngine.Serialization;

// Collision callbacks are only sent to the GameObject holding the Rigidbody, not to the agent root:
// this component, placed on every body part, forwards hurdle / floor / finish contacts to the agent.
// HurdleAgentBase adds it automatically to the body parts that do not have one.
public class BodyPartContact : MonoBehaviour
{
    [Tooltip("Agent to notify (searched in the parents if empty).")]
    [FormerlySerializedAs("agent")]
    [SerializeField] HurdleAgentBase owner;

    [Tooltip("Floor tag. Touching it ends the episode unless this body part is a foot.")]
    [FormerlySerializedAs("targetTag")]
    [SerializeField] string floorTag = HurdleAgentBase.FloorTag;

    [Tooltip("Checked = foot: this body part may touch the floor.")]
    [SerializeField] bool canTouchFloor;

    public HurdleAgentBase Agent
    {
        get => owner;
        set => owner = value;
    }

    public bool CanTouchFloor
    {
        get => canTouchFloor;
        set => canTouchFloor = value;
    }

    void Awake()
    {
        if (owner == null) owner = GetComponentInParent<HurdleAgentBase>();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (owner == null) return;

        var other = collision.collider;
        if (other.CompareTag(HurdleAgentBase.HurdleTag)) owner.OnHurdleHit();
        else if (other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
        else if (!canTouchFloor && other.CompareTag(floorTag)) owner.OnBodyPartTouchedFloor();
    }

    void OnTriggerEnter(Collider other)
    {
        if (owner != null && other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
    }
}
