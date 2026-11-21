using UnityEngine;

[DefaultExecutionOrder(-110)]
public class GroundContact : MonoBehaviour
{
    bool touchingThisStep;
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
        if (collision.collider.CompareTag(HurdleAgentBase.FloorTag))
        {
            touchingThisStep = true;
            IsTouchingGround = true;
        }
    }

    void OnCollisionStay(Collision collision)
    {
        if (collision.collider.CompareTag(HurdleAgentBase.FloorTag))
        {
            touchingThisStep = true;
            IsTouchingGround = true;
        }
    }
}