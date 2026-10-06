/* Accuracy policy banner — non-negotiable wording from README (Steps 7/8/10). */
export default function AccuracyBanner({ absoluteCrs }: { absoluteCrs?: boolean }) {
  return (
    <div className="banner">
      Relative accuracy (shape) ≠ absolute accuracy (position/scale). Ordinary GPS is
      metre-level — cm-level needs RTK/PPK. Points are metric-via-GPS-scale
      {absoluteCrs ? " in absolute CRS." : ", NOT absolute CRS (STEP 16 pending)."} Occluded
      surfaces are estimates, never measured.
    </div>
  );
}
