using System;
using UnityEngine;

// Bipedal ragdoll running the hurdles. Feet are the body parts named "foot*" (or carrying a GroundContact).
public class WalkerHurdleAgent : HurdleAgentBase
{
    [Header("Walker")]
    [Tooltip("Maximum drop (m) of the hips below their start height before the episode ends.")]
    [SerializeField] float fallMargin = 1.2f;

    protected override float FallMargin => fallMargin;
    protected override string BodyLayerName => "Walker";
    protected override string HurdleHeightParameter => "walker_hurdle_height";

    protected override bool IsFootPart(Rigidbody rb) =>
        rb.GetComponent<GroundContact>() != null
        || rb.name.IndexOf("foot", StringComparison.OrdinalIgnoreCase) >= 0;
}
