using UnityEngine;

public class BodyPartContact : MonoBehaviour
{
    [SerializeField] HurdleAgentBase owner;
    [SerializeField] bool canTouchFloor;

    Collider cachedFloorCollider;

    public HurdleAgentBase Agent { get => owner; set => owner = value; }
    public bool CanTouchFloor { get => canTouchFloor; set => canTouchFloor = value; }

    void Awake()
    {
        if (owner == null) owner = GetComponentInParent<HurdleAgentBase>();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (owner == null) return;
        var other = collision.collider;

        if (ReferenceEquals(other, cachedFloorCollider))
        {
            if (!canTouchFloor) owner.OnBodyPartTouchedFloor();
            return;
        }

        if (other.CompareTag(HurdleAgentBase.FloorTag))
        {
            cachedFloorCollider = other;
            if (!canTouchFloor) owner.OnBodyPartTouchedFloor();
        }
        else if (other.CompareTag(HurdleAgentBase.HurdleTag))
        {
            owner.OnHurdleHit(other);
        }
        else if (other.CompareTag(HurdleAgentBase.FinishTag))
        {
            owner.OnFinishReached();
        }
    }

    void OnCollisionStay(Collision collision)
    {
        // Fast-path: feet/shins stay in contact with the floor every step — avoid native marshaling & CompareTag.
        if (owner == null || canTouchFloor) return;

        var other = collision.collider;
        if (ReferenceEquals(other, cachedFloorCollider))
        {
            owner.OnBodyPartTouchedFloor();
            return;
        }

        if (other.CompareTag(HurdleAgentBase.FloorTag))
        {
            cachedFloorCollider = other;
            owner.OnBodyPartTouchedFloor();
        }
        else if (other.CompareTag(HurdleAgentBase.FinishTag))
        {
            owner.OnFinishReached();
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (owner != null && other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
    }
}
