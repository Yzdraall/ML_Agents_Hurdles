using UnityEngine;

[DefaultExecutionOrder(-110)]
public class GroundContact : MonoBehaviour
{
    bool touchingThisStep;
    Collider cachedFloorCollider;

    public bool IsTouchingGround { get; private set; }

    public void ResetContacts()
    {
        touchingThisStep = false;
        IsTouchingGround = false;
    }

    void FixedUpdate()
    {
        IsTouchingGround = touchingThisStep;
        touchingThisStep = false;
    }

    void OnCollisionEnter(Collision collision)
    {
        var other = collision.collider;
        if (ReferenceEquals(other, cachedFloorCollider) || other.CompareTag(HurdleAgentBase.FloorTag))
        {
            cachedFloorCollider = other;
            touchingThisStep = true;
            IsTouchingGround = true;
        }
    }

    void OnCollisionStay(Collision collision)
    {
        if (touchingThisStep) return;
        var other = collision.collider;
        if (ReferenceEquals(other, cachedFloorCollider) || other.CompareTag(HurdleAgentBase.FloorTag))
        {
            cachedFloorCollider = other;
            touchingThisStep = true;
            IsTouchingGround = true;
        }
    }
}
