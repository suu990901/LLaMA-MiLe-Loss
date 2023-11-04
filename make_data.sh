INTERMEDIATE_SCRATCH_PATH=proc_pile_all # save_path
PILE_DIR=pile #Your_Pile_Path
CACHE=cache
tokenizer=configs/tokenizer_models

# Process training data.
SPLIT=train

for PILE_DOMAIN in  "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles" ; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
python filter_domains.py --pile_path_dir ${PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain ${PILE_DOMAIN} --num_samples 102400000 --seed 111 --nproc 2 --subset ${SUBSET} --tokenizer ${tokenizer} &
done
wait
done

for PILE_DOMAIN in  "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
python filter_domains.py --pile_path_dir ${PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain ${PILE_DOMAIN} --num_samples 102400000 --seed 111 --nproc 2 --subset ${SUBSET} --tokenizer ${tokenizer} &
done
wait
done

# Process validation data.

SPLIT=validation

for PILE_DOMAIN in  "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles" ; do
for SUBSET in 00; do
python filter_domains.py --pile_path_dir ${PILE_DIR}  --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain ${PILE_DOMAIN} --num_samples 102400000 --seed 111 --nproc 2 --subset ${SUBSET} --tokenizer ${tokenizer} &
done
wait
done

for PILE_DOMAIN in  "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
for SUBSET in 00; do
python filter_domains.py --pile_path_dir ${PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain ${PILE_DOMAIN} --num_samples 102400000 --seed 111 --nproc 2 --subset ${SUBSET} --tokenizer ${tokenizer} &
done
wait
done
