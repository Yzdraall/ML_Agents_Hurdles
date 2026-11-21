using UnityEngine;

// Quadruped running the hurdles. Feet are the last segment of each leg (no Rigidbody below them).
public class CrawlerHurdleAgent : HurdleAgentBase
{
    [Header("Crawler")]
    [Tooltip("Maximum drop (m) of the body below its start height before the episode ends.")]
    [SerializeField] float fallMargin = 1.5f;
    [Tooltip("Joint target (degrees around X) applied at the start of each episode.")]
    [SerializeField] float neutralLegAngle = -30f;

    protected override float FallMargin => fallMargin;
    protected override string BodyLayerName => "Crawler";
    protected override string HurdleHeightParameter => "crawler_hurdle_height";
    protected override float DefaultHurdleHeight => 0.55f;
    protected override float MinSpawnClearance => 0.5f * (Root != null ? Root.lossyScale.y : transform.lossyScale.y);
    protected override float MinHeightAboveFloor => 0.5f;
    protected override Quaternion NeutralJointRotation => Quaternion.Euler(neutralLegAngle, 0f, 0f);

    protected override bool IsFootPart(Rigidbody rb) => rb.GetComponentsInChildren<Rigidbody>().Length == 1;
    protected override bool ShouldEndOnFloorContact() => !AnyFootTouchingGround || !IsUpright;
}
