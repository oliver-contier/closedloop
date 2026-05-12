# Generate odd-one-out triplets from in-silico FFA response patterns
# based on euclidean distance (cf. Methods: Generating simulated odd-one-out triplets).

transnames=("zvox")
simthresh="0.0"
ntrips="4_000_000"
subs=(1 2 3 4 all)

# use tmux
for ntrips in "${ntrips[@]}"; do
    for trans in "${transnames[@]}"; do
        for sub in "${subs[@]}"; do
            tmux new-session -d -s "tripletize_euclidean_sub-${sub}_trans-${trans}_ntrips-${ntrips}" "python tripletize_euclidean.py \
            --sub $sub \
            --simthresh $simthresh \
            --trans $trans \
            --ntrips $ntrips"
        done
    done
done

# use screen
# for trans in "${transnames[@]}"; do
#     for sub in "${subs[@]}"; do
#         screen -dms tripletize_euclidean_sub-${sub}_trans-${trans} python tripletize_euclidean.py \
#         --sub ${sub} \
#         --simthresh ${simthresh} \
#         --trans ${trans} \
#         --ntrips ${ntrips}
#     done
# done