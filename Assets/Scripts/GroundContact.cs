using UnityEngine;

public class GroundContact : MonoBehaviour
{
    public bool IsTouchingGround { get; private set; }
    private const string FLOORTAG = "floor";

    private void OnCollisionEnter(Collision col)
    {
        if (col.collider.CompareTag(FLOORTAG)) IsTouchingGround = true;
    }

    private void OnCollisionStay(Collision col)
    {
        if (col.collider.CompareTag(FLOORTAG)) IsTouchingGround = true;
    }

    private void OnCollisionExit(Collision col)
    {
        if (col.collider.CompareTag(FLOORTAG)) IsTouchingGround = false;
    }
}