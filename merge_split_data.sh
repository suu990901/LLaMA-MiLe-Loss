SPLIT="train" # Only the training data needs to be divided.

for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github"  ; do
python merge_split_data.py $SPLIT $PILE_DOMAIN &
done
wait

for PILE_DOMAIN in  "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles" "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
python merge_split_data.py $SPLIT $PILE_DOMAIN &
done
wait

for PILE_DOMAIN in "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" ; do
python merge_split_data.py $SPLIT $PILE_DOMAIN &
done
wait
