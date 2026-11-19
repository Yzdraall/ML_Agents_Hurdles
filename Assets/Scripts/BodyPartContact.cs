using UnityEngine;

public class BodyPartContact : MonoBehaviour
{
    [SerializeField] HurdleAgentBase owner;
    [SerializeField] bool canTouchFloor;

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

        if (other.CompareTag(HurdleAgentBase.HurdleTag)) owner.OnHurdleHit(other);
        else if (other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
        else if (!canTouchFloor && other.CompareTag(HurdleAgentBase.FloorTag)) owner.OnBodyPartTouchedFloor();
    }

    void OnCollisionStay(Collision collision)
    {
        if (owner == null) return;
        var other = collision.collider;

        if (other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
        else if (!canTouchFloor && other.CompareTag(HurdleAgentBase.FloorTag)) owner.OnBodyPartTouchedFloor();
    }

    void OnTriggerEnter(Collider other)
    {
        if (owner != null && other.CompareTag(HurdleAgentBase.FinishTag)) owner.OnFinishReached();
    }
}