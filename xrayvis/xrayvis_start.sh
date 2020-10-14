if [ ! -f all_words.feather ]; then
    wget --quiet https://d3uxfe7dw0hhy7.cloudfront.net/phonapps/xrayvis/all_words.feather
fi
if [ ! -f all_phones.feather ]; then
    wget --quiet https://d3uxfe7dw0hhy7.cloudfront.net/phonapps/xrayvis/all_phones.feather
fi
if [ ! -f file_opts.csv ]; then
    wget --quiet https://d3uxfe7dw0hhy7.cloudfront.net/phonapps/xrayvis/file_opts.csv
fi
if [ ! -f speaker_demographics1.csv ]; then
    wget --quiet https://d3uxfe7dw0hhy7.cloudfront.net/phonapps/xrayvis/speaker_demographics1.csv
fi
if [ ! -f speaker_demographics2.csv ]; then
    wget --quiet https://d3uxfe7dw0hhy7.cloudfront.net/phonapps/xrayvis/speaker_demographics2.csv
fi