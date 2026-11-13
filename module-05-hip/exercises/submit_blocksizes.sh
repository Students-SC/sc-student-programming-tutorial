#!/bin/bash
# ---------------------------------------------------------------------------
# submit_blocksizes.sh -- Exercise 3: compare block sizes on a GPU node
#
# First compile one program per block size and the odd-length version
# (README, Exercise 3, Step 1).
# Then submit:    sbatch submit_blocksizes.sh
#
# Runs every block size twice, then the odd-length version, and prints the
# results as one table.
# ---------------------------------------------------------------------------
#SBATCH --job-name=hip-blocksizes
#SBATCH -A TRN001 --reservation=sc26studentprogram --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --output=hip-blocksizes_%j.out
#SBATCH --error=hip-blocksizes_%j.err

BLOCK_SIZES="64 128 256 512 1024"

# Run one program and set TIME, BW, STATUS, MISMATCHES, and (on failure) MSG
# from its output.
run() {
    TIME=- BW=- STATUS=ERROR MISMATCHES=- MSG=""
    if [[ ! -x "$1" ]]; then
        STATUS=MISSING
        MSG="$1 not found: compile it first (Exercise 3, Step 1)"
        return
    fi
    local out rc
    out=$("$1" 2>&1)
    rc=$?
    TIME=$(awk '/Time:/ {print $2}' <<< "$out")
    BW=$(awk '/Effective bandwidth:/ {print $3}' <<< "$out")
    MISMATCHES=$(awk -F': *' '/Mismatches:/ {print $2}' <<< "$out")
    STATUS=$(awk '/Status:/ {print $2}' <<< "$out")
    [[ -z "$STATUS" ]] && STATUS=ERROR
    if [[ $rc -ne 0 ]]; then
        MSG="$1: $(grep -m1 -i "error" <<< "$out")"
    fi
}

echo "Block-size comparison -- Job $SLURM_JOB_ID on $(hostname)"
echo "Each time is the average of 10 kernel launches; each block size runs twice."
echo ""

declare -A T B S
ERRORS=()
for pass in 1 2; do
    for b in $BLOCK_SIZES; do
        run ./vector_scale_$b
        T[$pass,$b]=${TIME:--}
        B[$pass,$b]=${BW:--}
        S[$pass,$b]=$STATUS
        [[ -n "$MSG" ]] && ERRORS+=("run $pass, $MSG")
    done
done

printf "%-10s | %-17s | %-23s | %s\n" "" "Time (ms)" "Effective bw (GB/s)" "Status"
printf "%-10s | %-8s %-8s | %-11s %-11s | %-8s %s\n" \
       "Block size" "Run 1" "Run 2" "Run 1" "Run 2" "Run 1" "Run 2"
for b in $BLOCK_SIZES; do
    printf "%-10s | %-8s %-8s | %-11s %-11s | %-8s %s\n" "$b" \
           "${T[1,$b]}" "${T[2,$b]}" "${B[1,$b]}" "${B[2,$b]}" "${S[1,$b]}" "${S[2,$b]}"
done

echo ""
run ./vector_scale_oddN
echo "Odd length (vector_scale_oddN): $STATUS, mismatches: $MISMATCHES"
[[ -n "$MSG" ]] && ERRORS+=("$MSG")

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "Errors:"
    printf "  %s\n" "${ERRORS[@]}"
fi
