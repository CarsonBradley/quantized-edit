#!/bin/bash
# calibrate.sh - show that the prefill-vs-decode consistency metric of gpu_validate.sh step 5 (edit norm ratio and
# correlation, edit_delta.py) detects an edit that is applied only partly. Five --dump-ref runs of the same 128
# WikiText tokens (about 10-15 s each), reusing gpu_validate.sh's dumps (flag off/on at slices 64 and 1):
#   S=0.9 at slice 64 and slice 1   -> expected norm ratio ~0.9 against S=1 (a uniform 10% under-application)
#   S=0.5 at slice 64               -> expected ~0.5 (linearity)
#   DIRS_NOEMB at slice 64 / 1      -> embedding site's real direction swapped for the min-energy direction of the
#                                      embedding table (make_dirs_noemb.py): removes the real embedding edit and
#                                      adds a smaller unrelated projection
#
# usage: ENGINE=... PACK=... DIRS=... DIRS_NOEMB=... EVAL_DATA=... bash engine/validate/calibrate.sh RUN_DIR
#   RUN_DIR is gpu_validate.sh's OUT_DIR; writes RUN_DIR/CALIBRATION.md.
#   GUARD (optional) as in gpu_validate.sh; exit 75/76 from it -> retry every 60 s for up to 90 min.
set -u
: "${ENGINE:?set ENGINE}" "${PACK:?set PACK}" "${DIRS:?set DIRS}" "${DIRS_NOEMB:?set DIRS_NOEMB}" "${EVAL_DATA:?set EVAL_DATA}"
[ $# -ge 1 ] || { echo "usage: calibrate.sh RUN_DIR (see the header)" >&2; exit 2; }
T="$(cd "$(dirname "$0")" && pwd)"
R=$1; D=$R/dumps
GUARD=${GUARD:-}
IDS=$(python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));print(",".join(map(str,d["windows"][2][:128])))' "$EVAL_DATA/wikitext2_windows.json")
for f in off_128_s64 on_128_s64 off_128_s1 on_128_s1; do [ -e "$D/$f.bin" ] || { echo "missing $D/$f.bin (run gpu_validate.sh first)"; exit 1; }; done
run() {
    local name=$1 out=$2; shift 2
    [ -e "$out" ] && { echo "$name: exists, skipping"; return 0; }
    for i in $(seq 1 90); do
        if [ -n "$GUARD" ]; then
            $GUARD --name "abl-cal-$name" --need-gb 18 -- "$@" > "$R/cal_$name.log" 2>&1
        else
            "$@" > "$R/cal_$name.log" 2>&1
        fi
        rc=$?
        [ -z "$GUARD" ] && break
        [ $rc -ne 75 ] && [ $rc -ne 76 ] && break
        sleep 60
    done
    echo "$(date +%H:%M:%S) $name rc=$rc"
    [ $rc -eq 0 ] || { tail -5 "$R/cal_$name.log"; exit 1; }
}
E="$ENGINE"
run s09-s64 "$D/on09_128_s64.bin" env STRATA_INK_EVAL_LOGIT_SLICE=64 "$E" --dump-ref "$D/on09_128_s64.bin" --native "$PACK" --max-context 8192 --prompt-ids "$IDS" --ablate "$DIRS" --ablate-scale 0.9
run s09-s1  "$D/on09_128_s1.bin"  env STRATA_INK_EVAL_LOGIT_SLICE=1  "$E" --dump-ref "$D/on09_128_s1.bin"  --native "$PACK" --max-context 8192 --prompt-ids "$IDS" --ablate "$DIRS" --ablate-scale 0.9
run s05-s64 "$D/on05_128_s64.bin" env STRATA_INK_EVAL_LOGIT_SLICE=64 "$E" --dump-ref "$D/on05_128_s64.bin" --native "$PACK" --max-context 8192 --prompt-ids "$IDS" --ablate "$DIRS" --ablate-scale 0.5
run noemb-s64 "$D/noemb_128_s64.bin" env STRATA_INK_EVAL_LOGIT_SLICE=64 "$E" --dump-ref "$D/noemb_128_s64.bin" --native "$PACK" --max-context 8192 --prompt-ids "$IDS" --ablate "$DIRS_NOEMB"
run noemb-s1  "$D/noemb_128_s1.bin"  env STRATA_INK_EVAL_LOGIT_SLICE=1  "$E" --dump-ref "$D/noemb_128_s1.bin"  --native "$PACK" --max-context 8192 --prompt-ids "$IDS" --ablate "$DIRS_NOEMB"
ED() { uv run -q --no-project --with numpy python -I "$T/edit_delta.py" "$@" | python3 -I -c 'import json,sys;d=json.load(sys.stdin);print("norm ratio %.3f | corr %.3f | edit KL A %.4f B %.4f" % (d["median_norm_ratio_B_over_A"],d["median_corr"],d["kl_edit_pathA"],d["kl_edit_pathB"]))'; }
{
  echo "# Step-5 metric calibration ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "Reference is the full edit (S=1) on the prefill path (slice 64): D_A = on_s64 - off_s64."
  echo
  echo "| path B | result |"
  echo "|---|---|"
  echo "| full edit, decode path (slice 1), from gpu_validate.sh | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s1.bin $D/on_128_s1.bin) |"
  echo "| S=0.9, same prefill path | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s64.bin $D/on09_128_s64.bin) |"
  echo "| S=0.9, decode path | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s1.bin $D/on09_128_s1.bin) |"
  echo "| S=0.5, same prefill path | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s64.bin $D/on05_128_s64.bin) |"
  echo "| embedding site broken, prefill path | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s64.bin $D/noemb_128_s64.bin) |"
  echo "| embedding site broken on the decode path only | $(ED $D/off_128_s64.bin $D/on_128_s64.bin $D/off_128_s1.bin $D/noemb_128_s1.bin) |"
} | tee "$R/CALIBRATION.md"
