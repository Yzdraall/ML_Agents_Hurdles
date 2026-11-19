using System;
using UnityEngine;

// Bipedal ragdoll running the hurdles. Feet are the body parts named "foot*" (or carrying a GroundContact).
public class WalkerHurdleAgent : HurdleAgentBase
{
    [Header("Walker")]
    [Tooltip("Maximum drop (m) of the hips below their start height before the episode ends.")]
    [SerializeField] float fallMargin = 2.2f;

    protected override float FallMargin => fallMargin;
    protected override string BodyLayerName => "Walker";
    protected override string HurdleHeightParameter => "walker_hurdle_height";
    protected override float MinSpawnClearance => 0.02f;
    protected override float MinHeightAboveFloor => 0.6f;

    protected override bool IsFootPart(Rigidbody rb) =>
        rb.GetComponent<GroundContact>() != null
        || rb.name.IndexOf("foot", StringComparison.OrdinalIgnoreCase) >= 0;

    protected override bool CanTouchFloor(Rigidbody rb) =>
        IsFootPart(rb) || rb.name.IndexOf("shin", StringComparison.OrdinalIgnoreCase) >= 0;
}
