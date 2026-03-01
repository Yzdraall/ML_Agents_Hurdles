using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target; // Glissez le Crawler ici dans l'Inspecteur
    public Vector3 offset = new Vector3(0, 5, -10); // Distance par rapport à l'agent
    public float smoothSpeed = 0.125f;

    void LateUpdate()
    {
        if (target != null)
        {
            Vector3 desiredPosition = target.position + offset;
            Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed);
            transform.position = smoothedPosition;

            transform.LookAt(target); // La caméra regarde toujours l'agent
        }
    }
}