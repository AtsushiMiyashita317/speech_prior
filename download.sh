subsets=(
    dev-clean
    train-clean-100
)


for subset in "${subsets[@]}"; do
    echo "Downloading and extracting ${subset}..."
    # データ保存先を用意
    mkdir -p data/librispeech
    cd data/librispeech

    wget http://www.openslr.org/resources/12/${subset}.tar.gz
    tar -xvzf ${subset}.tar.gz
done
