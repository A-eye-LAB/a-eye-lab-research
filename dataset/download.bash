mkdir -p data/real_data
mkdir -p data/preprocessed

# Download real data
huggingface-cli download a-eyelab/real-data-crop --repo-type dataset --local-dir ./data/tmp
mv ./data/tmp/0 ./data/tmp/1 ./data/real_data

# Download preprocessed data
huggingface-cli download a-eyelab/preprocessed --repo-type dataset --local-dir ./data/tmp
mv ./data/tmp/0 ./data/tmp/1 ./data/preprocessed
rm -rf ./data/tmp