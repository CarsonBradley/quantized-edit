#!/bin/bash
# gpu_validate.sh - GPU validation of the --ablate engine build: flag-off parity with the stock binary, scale-0
# identity, KL of the edit, prefill-vs-decode consistency, speculative decoding with the flag on, coherence, a
# 5-prompt refusal smoke test and (optionally) speed benches. Steps and pass rules: engine/validate/RESULTS.md.
#
# usage: ENGINE=... SHIP=... PACK=... DIRS=... MODEL_DIR=... EVAL_DATA=... bash engine/validate/gpu_validate.sh OUT_DIR
#        SKIP_BENCH=1 ...    skip the speed benches (recorded as PENDING)
#        BENCH_ONLY=1 ...    only the speed benches (e.g. later, when the machine is quiet)
#
#   ENGINE     the patched strata-inkling (kld fix + ablate patch)
#   SHIP       the stock strata-inkling 0.1.65 binary, for the parity checks
#   PACK       Inkling-2-Mini-SQ-15.strata (the original pack)
#   DIRS       the direction file from engine/make_dirs.py
#   MODEL_DIR  the pack's Hugging Face folder: tokenizer.json, chat_template.jinja, tools/prompts/p_short.txt
#   EVAL_DATA  the folder eval/prep_data.py writes (wikitext2_windows.json, harmful.jsonl)
#   GUARD      optional: a command prefix that wraps every engine run, e.g. a memory/GPU lock. It is called as
#              $GUARD --name NAME --need-gb 18 [--exclusive] -- COMMAND...
#              and may exit 75 (busy: the script waits 60 s and retries), 76 (refused: abort, or PENDING for the
#              --exclusive benches) or 99 (killed: abort); any other code is the command's own. Unset: run directly.
#
# Results: OUT_DIR/RESULTS.md (PASS/FAIL/INFO/PENDING per step + key numbers), OUT_DIR/results.jsonl, a log per run.
# Non-bench runs use --max-context 8192 and short inputs; benches use 32768.
set -u

: "${ENGINE:?set ENGINE}" "${SHIP:?set SHIP}" "${PACK:?set PACK}" "${DIRS:?set DIRS}" "${MODEL_DIR:?set MODEL_DIR}" "${EVAL_DATA:?set EVAL_DATA}"
[ $# -ge 1 ] || { echo "usage: gpu_validate.sh OUT_DIR (see the header)" >&2; exit 2; }
T="$(cd "$(dirname "$0")" && pwd)"                 # this folder (helpers)
EVAL="$(cd "$T/../../eval" && pwd)"               # the repo's eval/ (refusal classifier)
MDIR=$MODEL_DIR
ABL=$ENGINE
GUARD=${GUARD:-}
OUT=$1
CTX=8192
mkdir -p "$OUT"
RES=$OUT/RESULTS.md
JL=$OUT/results.jsonl
PY="uv run -q --no-project --with numpy --with jinja2 --with tokenizers python -I"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log" >&2; }
result() {  # result STEP STATUS DETAILS
    echo "| $1 | **$2** | $3 |" >> "$RES"
    python3 -I -c 'import json,sys;print(json.dumps({"step":sys.argv[1],"status":sys.argv[2],"details":sys.argv[3]}))' "$1" "$2" "$3" >> "$JL"
    log "RESULT $1: $2  $3"
}
abort() { result "ABORT" "FAIL" "$1"; log "ABORTING: $1"; exit 1; }

# guarded NAME EXCL(0|1) LOGFILE -- cmd ...   -> returns the command's exit code; handles 75/76/99 from $GUARD
guarded() {
    local name=$1 excl=$2 lf=$3; shift 4
    local xflag=""; [ "$excl" = 1 ] && xflag="--exclusive"
    while true; do
        log "run [$name] $*"
        if [ -n "$GUARD" ]; then
            $GUARD --name "abl-$name" --need-gb 18 $xflag -- "$@" > "$lf" 2>&1
        else
            "$@" > "$lf" 2>&1
        fi
        local rc=$?
        [ -z "$GUARD" ] && { [ $rc -ne 0 ] && log "  exit $rc (see $lf)"; return $rc; }
        if [ $rc -eq 75 ]; then log "  lock held by another worker; retry in 60 s"; sleep 60; continue; fi
        if [ $rc -eq 76 ]; then
            log "  preflight refused: $(grep -A3 REFUSED "$lf" | tr '\n' ' ')"
            [ "$excl" = 1 ] && return 76
            abort "guard refused $name (see $lf)"
        fi
        if [ $rc -eq 99 ]; then abort "guard killed $name (see $lf)"; fi
        [ $rc -ne 0 ] && log "  exit $rc (see $lf)"
        return $rc
    done
}

ids_window() {  # ids_window WINDOW_INDEX N  -> comma-separated ids (WikiText-2 test windows from eval/prep_data.py)
    python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));w=d["windows"][int(sys.argv[2])][:int(sys.argv[3])];print(",".join(map(str,w)))' \
        "$EVAL_DATA/wikitext2_windows.json" "$1" "$2"
}
cmp_ref() { $PY "$T/compare_ref.py" "$@"; }
jget() { python3 -I -c 'import json,sys;d=json.loads(sys.argv[1]);v=d[sys.argv[2]];print(v if not isinstance(v,float) else ("%.6g"%v))' "$1" "$2"; }
same_ids() { python3 -I -c 'import json,sys;a=json.load(open(sys.argv[1]))["ids"];b=json.load(open(sys.argv[2]))["ids"];n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)));print("identical" if a==b else "first diff at token %d (len %d vs %d)"%(n,len(a),len(b)))' "$1" "$2"; }

for f in "$SHIP" "$ABL" "$DIRS" "$PACK" "$MDIR/tokenizer.json" "$MDIR/chat_template.jinja" "$MDIR/tools/prompts/p_short.txt" \
         "$EVAL_DATA/wikitext2_windows.json" "$EVAL_DATA/harmful.jsonl"; do
    [ -e "$f" ] || abort "missing $f"
done
{
  echo "# GPU validation of the --ablate build"
  echo
  echo "- started $(date '+%Y-%m-%d %H:%M:%S'), out dir \`$OUT\`"
  echo "- shipped: \`$SHIP\` ($(shasum -a 256 "$SHIP" | cut -c1-12)); ablate build: \`$ABL\` ($(shasum -a 256 "$ABL" | cut -c1-12)); DIRS.bin $(shasum -a 256 "$DIRS" | cut -c1-12)"
  echo "- guard: ${GUARD:-none}"
  echo
  echo "| step | status | details |"
  echo "|---|---|---|"
} > "$RES"
: > "$JL"

if [ "${BENCH_ONLY:-0}" = 1 ]; then STEPS_BEFORE_BENCH=0; else STEPS_BEFORE_BENCH=1; fi
IDS128=$(ids_window 2 128)
WIN0=$(ids_window 0 512)
WIN1=$(ids_window 1 512)
D=$OUT/dumps; mkdir -p "$D"
if [ "$STEPS_BEFORE_BENCH" = 1 ]; then   # BENCH_ONLY=1 skips steps 1-8

# ---------------------------------------------------------------- 1. 64-row logits bug
guarded ship-dump128 0 "$OUT/1_ship_dump128.log" -- "$SHIP" --dump-ref "$D/ship_128.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" || abort "shipped --dump-ref failed"
guarded off-dump128 0 "$OUT/1_off_dump128.log" -- "$ABL" --dump-ref "$D/off_128_s64.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" || abort "ablate-build --dump-ref failed"
J=$(cmp_ref "$D/ship_128.bin" "$D/off_128_s64.bin"); echo "$J" > "$OUT/1_cmp.json"
ZS=$(jget "$J" zero_rows_A_from64); ZO=$(jget "$J" zero_rows_B_from64)
if [ "$ZO" = 0 ]; then
    result "1 64-row logits bug" "PASS" "shipped: $ZS/64 all-zero rows in rows 64-127 ($([ "$ZS" = 64 ] && echo 'bug confirmed in stock' || echo 'bug NOT reproduced in stock')); fixed build: 0 zero rows"
else
    result "1 64-row logits bug" "FAIL" "fixed build still has $ZO zero rows in rows 64-127 (shipped $ZS)"
fi

# ---------------------------------------------------------------- 2. flag-off parity vs shipped
J=$(cmp_ref "$D/ship_128.bin" "$D/off_128_s64.bin" --rows 0:64); echo "$J" > "$OUT/2_cmp_rows0_63.json"
MX=$(jget "$J" max_abs_diff); BW=$(jget "$J" bitwise_equal); T1=$(jget "$J" top1_agree)
if python3 -I -c "import sys;sys.exit(0 if float('$MX')<=1e-3 and float('$T1')==1.0 else 1)"; then
    result "2a flag-off logits vs shipped (rows 0-63)" "PASS" "bitwise_equal=$BW max|dlogit|=$MX top1=$T1"
else
    result "2a flag-off logits vs shipped (rows 0-63)" "FAIL" "bitwise_equal=$BW max|dlogit|=$MX top1=$T1"
fi
for dr in off auto; do
    guarded "ship-serve-$dr" 0 "$OUT/2_ship_serve_$dr.log" -- $PY "$T/serve_ids.py" "$SHIP" "$PACK" "$MDIR" "$MDIR/tools/prompts/p_short.txt" 200 "$OUT/2_ship_serve_$dr.json" --draft $dr --max-context $CTX || abort "shipped serve ($dr) failed"
    guarded "off-serve-$dr" 0 "$OUT/2_off_serve_$dr.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$MDIR/tools/prompts/p_short.txt" 200 "$OUT/2_off_serve_$dr.json" --draft $dr --max-context $CTX || abort "ablate-build serve ($dr) failed"
    S=$(same_ids "$OUT/2_ship_serve_$dr.json" "$OUT/2_off_serve_$dr.json")
    [ "$S" = identical ] && result "2b flag-off greedy tokens vs shipped (--draft $dr, 200 tok)" "PASS" "$S" \
                         || result "2b flag-off greedy tokens vs shipped (--draft $dr, 200 tok)" "FAIL" "$S"
done
S=$(same_ids "$OUT/2_ship_serve_off.json" "$OUT/2_ship_serve_auto.json")
result "2c (info) stock: DSpark vs plain greedy" "$([ "$S" = identical ] && echo PASS || echo INFO)" "$S"

# ---------------------------------------------------------------- 3. --ablate-scale 0 == flag off
guarded s0-dump128 0 "$OUT/3_s0_dump128.log" -- "$ABL" --dump-ref "$D/s0_128.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" --ablate "$DIRS" --ablate-scale 0 || abort "S=0 dump failed"
J=$(cmp_ref "$D/off_128_s64.bin" "$D/s0_128.bin"); echo "$J" > "$OUT/3_cmp.json"
MX=$(jget "$J" max_abs_diff); BW=$(jget "$J" bitwise_equal); T1=$(jget "$J" top1_agree)
python3 -I -c "import sys;sys.exit(0 if float('$MX')<=1e-3 and float('$T1')==1.0 else 1)" \
    && result "3a --ablate-scale 0 logits == flag off (128 rows)" "PASS" "bitwise_equal=$BW max|dlogit|=$MX" \
    || result "3a --ablate-scale 0 logits == flag off (128 rows)" "FAIL" "bitwise_equal=$BW max|dlogit|=$MX top1=$T1"
guarded s0-serve 0 "$OUT/3_s0_serve.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$MDIR/tools/prompts/p_short.txt" 200 "$OUT/3_s0_serve_off.json" --draft off --max-context $CTX --ablate "$DIRS" --ablate-scale 0 || abort "S=0 serve failed"
S=$(same_ids "$OUT/2_off_serve_off.json" "$OUT/3_s0_serve_off.json")
[ "$S" = identical ] && result "3b --ablate-scale 0 greedy tokens == flag off" "PASS" "$S" || result "3b --ablate-scale 0 greedy tokens == flag off" "FAIL" "$S"

# ---------------------------------------------------------------- 4. KL ablated vs stock on plain text (wikitext-2 test)
KLS=""
for wi in 0 1; do
    IDS=$WIN0; [ $wi = 1 ] && IDS=$WIN1
    guarded "off-dump-w$wi" 0 "$OUT/4_off_w$wi.log" -- "$ABL" --dump-ref "$D/off_w$wi.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS" || abort "dump off w$wi failed"
    guarded "on-dump-w$wi" 0 "$OUT/4_on_w$wi.log" -- "$ABL" --dump-ref "$D/on_w$wi.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS" --ablate "$DIRS" || abort "dump on w$wi failed"
    J=$(cmp_ref "$D/off_w$wi.bin" "$D/on_w$wi.bin"); echo "$J" > "$OUT/4_cmp_w$wi.json"
    KLS="$KLS w$wi: KL=$(jget "$J" mean_kl) top1=$(jget "$J" top1_agree) ppl $(jget "$J" ppl_A)->$(jget "$J" ppl_B) zero_rows=$(jget "$J" zero_rows_A)/$(jget "$J" zero_rows_B);"
done
guarded on-kldeval-w0 0 "$OUT/4_kldeval_w0.log" -- "$ABL" --kld-eval "$D/off_w0.bin" --native "$PACK" --max-context $CTX --ablate "$DIRS" || abort "kld-eval failed"
KE=$(grep -h "\[kld-eval\]" "$OUT/4_kldeval_w0.log" | tail -1)
KLOK=$(python3 -I -c '
import json,sys
ks=[json.load(open(f))["mean_kl"] for f in sys.argv[1:]]
z=[json.load(open(f))["zero_rows_A"]+json.load(open(f))["zero_rows_B"] for f in sys.argv[1:]]
m=sum(ks)/len(ks); print("PASS" if 1e-4 < m < 0.05 and sum(z)==0 else "FAIL", "%.5f"%m)' "$OUT/4_cmp_w0.json" "$OUT/4_cmp_w1.json")
result "4 KL(stock || ablated), 2 x 512-token wikitext-2 windows" "${KLOK%% *}" "mean KL ${KLOK#* } (AtomicChat BF16: 0.008);$KLS engine: $KE"

# ---------------------------------------------------------------- 5. prefill vs decode path consistency (slices 1/8/12/64)
# The same 128 positions computed through: slice 64 (prefill: MPP GEMMs + MPP attention + prefill sconv),
# 12 (MPP MoE + MPP trunk + multi-slot sconv path), 8 (non-MPP, verify-ring path, fused-decode replacement M3),
# 1 (pure decode: Q8/BF16 GEMV, M3). A missed insertion point makes the flag-on spread much larger than flag-off.
guarded on-dump128 0 "$OUT/5_on_s64.log" -- "$ABL" --dump-ref "$D/on_128_s64.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" --ablate "$DIRS" || abort "on s64 dump failed"
J=$(cmp_ref "$D/off_128_s64.bin" "$D/on_128_s64.bin"); EDIT_KL=$(jget "$J" mean_kl)
for sl in 1 8 12; do
    guarded "off-dump128-s$sl" 0 "$OUT/5_off_s$sl.log" -- env STRATA_INK_EVAL_LOGIT_SLICE=$sl "$ABL" --dump-ref "$D/off_128_s$sl.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" || abort "off s$sl dump failed"
    guarded "on-dump128-s$sl" 0 "$OUT/5_on_s$sl.log" -- env STRATA_INK_EVAL_LOGIT_SLICE=$sl "$ABL" --dump-ref "$D/on_128_s$sl.bin" --native "$PACK" --max-context $CTX --prompt-ids "$IDS128" --ablate "$DIRS" || abort "on s$sl dump failed"
    JO=$(cmp_ref "$D/off_128_s64.bin" "$D/off_128_s$sl.bin"); JN=$(cmp_ref "$D/on_128_s64.bin" "$D/on_128_s$sl.bin")
    echo "$JO" > "$OUT/5_cmp_off_s$sl.json"; echo "$JN" > "$OUT/5_cmp_on_s$sl.json"
    # Criterion (revised after run1): the cross-path spread is heavy-tailed fp noise (Q8 GEMV vs FP16 MPP; at slice 12
    # flag-off spread was 2.4x flag-on), so the decisive test is whether path B applies the SAME edit as path A:
    # D = on - off (log-softmax) per path; median norm ratio |D_B|/|D_A| in [0.95, 1.05], median corr >= 0.88.
    # Calibrated (engine/validate/CALIBRATION.md): correct paths 0.992-1.001 / 0.917-0.937; S=0.9 -> 0.922-0.925 (fails
    # ratio); S=0.5 -> 0.694; embedding site broken -> corr 0.78-0.83 (fails corr).
    JD=$($PY "$T/edit_delta.py" "$D/off_128_s64.bin" "$D/on_128_s64.bin" "$D/off_128_s$sl.bin" "$D/on_128_s$sl.bin"); echo "$JD" > "$OUT/5_delta_s$sl.json"
    V=$(python3 -I -c '
import json,sys
o,n,e,d=json.loads(sys.argv[1]),json.loads(sys.argv[2]),float(sys.argv[3]),json.loads(sys.argv[4])
ok = 0.95 <= d["median_norm_ratio_B_over_A"] <= 1.05 and d["median_corr"] >= 0.88 and n["mean_kl"] <= max(4*o["mean_kl"], 2e-4) and n["zero_rows_B"]==0
print("PASS" if ok else "FAIL", "edit applied on this path: norm ratio %.3f corr %.3f (edit KL %.2e vs %.2e) | cross-path off: KL %.2e max|d| %.3g top1 %.3f | on: KL %.2e max|d| %.3g top1 %.3f" % (
  d["median_norm_ratio_B_over_A"], d["median_corr"], d["kl_edit_pathB"], e,
  o["mean_kl"],o["max_abs_diff"],o["top1_agree"],n["mean_kl"],n["max_abs_diff"],n["top1_agree"]))' "$JO" "$JN" "$EDIT_KL" "$JD")
    result "5 slice $sl vs slice 64 (same 128 positions)" "${V%% *}" "${V#* }"
done

# ---------------------------------------------------------------- 6/7. DSpark on vs off with the flag on; coherence
guarded on-serve-off 0 "$OUT/6_on_serve_off.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$MDIR/tools/prompts/p_short.txt" 200 "$OUT/6_on_serve_off.json" --draft off --max-context $CTX --ablate "$DIRS" || abort "on serve (draft off) failed"
guarded on-serve-auto 0 "$OUT/6_on_serve_auto.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$MDIR/tools/prompts/p_short.txt" 200 "$OUT/6_on_serve_auto.json" --draft auto --max-context $CTX --ablate "$DIRS" || abort "on serve (draft auto) failed"
S=$(same_ids "$OUT/6_on_serve_off.json" "$OUT/6_on_serve_auto.json")
S0=$(same_ids "$OUT/2_off_serve_off.json" "$OUT/2_off_serve_auto.json")
if [ "$S" = identical ]; then result "6 flag on: DSpark (tree/verify paths) vs plain greedy" "PASS" "$S (flag off: $S0)"
elif [ "$S0" != identical ]; then result "6 flag on: DSpark (tree/verify paths) vs plain greedy" "INFO" "$S; stock also differs ($S0), so not attributable to the patch"
else result "6 flag on: DSpark (tree/verify paths) vs plain greedy" "FAIL" "$S (flag off identical)"; fi
C=$(python3 -I -c '
import json,sys
r=json.load(open(sys.argv[1])); t=r["text"]; d=(r["done"] or "").split()
ok = len(r["ids"])>=50 and ("def " in t) and "nonfinite" not in (r["done"] or "")
print("PASS" if ok else "FAIL", "gen %d tok, stop=%s, python def present=%s, differs from flag-off=%s" % (len(r["ids"]), d[-1] if d else "?", "def " in t,
      r["ids"]!=json.load(open(sys.argv[2]))["ids"]))' "$OUT/6_on_serve_off.json" "$OUT/2_off_serve_off.json")
result "7 coherence, flag on (p_short: Python CSV function)" "${C%% *}" "${C#* } (text in 6_on_serve_off.json)"

# ---------------------------------------------------------------- 8. refusal smoke, 5 harmful prompts, flag off vs on
python3 -I -c 'import json,sys;print(json.dumps([json.loads(l)["prompt"] for l in open(sys.argv[1]) if l.strip()][:5]))' "$EVAL_DATA/harmful.jsonl" > "$OUT/8_prompts.json"
guarded smoke-off 0 "$OUT/8_off.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$OUT/8_prompts.json" 320 "$OUT/8_off.json" --many --effort none --draft off --max-context $CTX || abort "smoke off failed"
guarded smoke-on 0 "$OUT/8_on.log" -- $PY "$T/serve_ids.py" "$ABL" "$PACK" "$MDIR" "$OUT/8_prompts.json" 320 "$OUT/8_on.json" --many --effort none --draft off --max-context $CTX --ablate "$DIRS" || abort "smoke on failed"
RO=$($PY "$T/smoke_refusal.py" "$OUT/8_off.json" "$EVAL" "$MDIR"); RN=$($PY "$T/smoke_refusal.py" "$OUT/8_on.json" "$EVAL" "$MDIR")
echo "$RO" > "$OUT/8_off_cls.json"; echo "$RN" > "$OUT/8_on_cls.json"
NO=$(jget "$RO" refusals); NN=$(jget "$RN" refusals)
CLS=$(python3 -I -c 'import json,sys;print(" | ".join("%s:%s" % (k, ",".join(p["answer_class"] for p in json.loads(s)["per_prompt"])) for k,s in (("off",sys.argv[1]),("on",sys.argv[2]))))' "$RO" "$RN")
result "8 refusal smoke (first 5 harmful, effort none, 320 tok)" "$([ "$NN" -lt "$NO" ] && echo PASS || echo INFO)" "refusals off $NO/5, on $NN/5; $CLS"

fi   # end of steps 1-8

# ---------------------------------------------------------------- 9. benches (exclusive)
bench() {  # bench LABEL BIN PLEN [extra...]
    local label=$1 bin=$2 plen=$3; shift 3
    local lf="$OUT/9_bench_${label}_${plen}.log"
    guarded "bench-$label-$plen" 1 "$lf" -- "$bin" --bench --native "$PACK" --max-context 32768 --prompt-len "$plen" --gen 128 --draft off "$@"
    local rc=$?
    if [ $rc -eq 76 ]; then echo "PENDING"; return; fi
    [ $rc -ne 0 ] && { echo "FAILED"; return; }
    grep -h "^\[bench\]" "$lf" | sed -E 's/.*prefill=([0-9.]+) ppts.*decode=([0-9.]+) tps agg.*/\1 \2/'
}
if [ "${SKIP_BENCH:-0}" = 1 ]; then
    result "9 bench" "PENDING" "SKIP_BENCH=1"
else
    BTAB=""; PEND=0
    for plen in 6630 28156; do
        for rep in 1 2; do
            for cfg in ship off on; do
                case $cfg in
                    ship) r=$(bench "ship-r$rep" "$SHIP" $plen) ;;
                    off)  r=$(bench "off-r$rep" "$ABL" $plen) ;;
                    on)   r=$(bench "on-r$rep" "$ABL" $plen --ablate "$DIRS") ;;
                esac
                [ "$r" = PENDING ] && { PEND=1; break 3; }
                BTAB="$BTAB $cfg/$plen/r$rep: ${r};"
            done
        done
    done
    if [ $PEND = 1 ]; then result "9 bench (--draft off, 6630 & 28156, x2)" "PENDING" "the guard refused an exclusive run; partial:$BTAB"
    else result "9 bench (--draft off, 6630 & 28156, x2)" "INFO" "prefill_tok/s decode_tok/s:$BTAB"; fi
fi

echo >> "$RES"; echo "- finished $(date '+%Y-%m-%d %H:%M:%S')" >> "$RES"
log "done -> $RES"
