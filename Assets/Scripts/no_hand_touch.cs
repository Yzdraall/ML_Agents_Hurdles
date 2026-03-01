using UnityEngine;
using Unity.MLAgents;
using TMPro;
using Unity.VisualScripting;

public class no_hand_touch : MonoBehaviour
{
    public WalkerAgent agent; 
    public string targetTag = "floor"; 
    void Start()
    {
        //si l'agent est pas trouvé
        if (agent == null)
        {
            //il va le chercher
            agent = GetComponentInParent<WalkerAgent>();
        }
    }

    void OnCollisionStay(Collision collision)
    {
        //si on ne trouve toujours pas l'agent on ne fait rien
        if (agent == null) return; 

        if (collision.gameObject.CompareTag("floor"))
        {
            agent.AddReward(-0.5f);
            agent.EndEpisode();
             
        }
    }
}