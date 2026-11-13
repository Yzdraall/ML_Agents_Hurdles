using UnityEngine;

// Tracks whether a foot touches the floor. Uses a contact counter so that leaving one floor collider
// while still touching another does not report "in the air".
public class GroundContact : MonoBehaviour
{
    int floorContacts;

    public bool IsTouchingGround => floorContacts > 0;

    // Called by the agent at the start of each episode (the body is teleported, pending contacts are stale).
    public void ResetContacts() => floorContacts = 0;

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag(HurdleAgentBase.FloorTag)) floorContacts++;
    }

    void OnCollisionExit(Collision collision)
    {
        if (collision.collider.CompareTag(HurdleAgentBase.FloorTag)) floorContacts = Mathf.Max(0, floorContacts - 1);
    }
}
