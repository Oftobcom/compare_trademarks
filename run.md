python compare_trademarks_v10.py --new test_logos/adidas2.jpg --collections logos --embedding none --visualize --html --recursive

python compare_trademarks_v10.py --new test_logos/adidas2.jpg --collections logos --embedding none --visualize --html --recursive

python compare_trademarks_v10.py --new test_logos/adidas2.jpg --collections logos --embedding both --semantic-weight 0.25 --cache --visualize --html --recursive

python compare_trademarks_v10.py --new test_logos/citroen.jpg --collections logos --embedding both --semantic-weight 0.25 --cache --visualize --html

python compare_trademarks_v10.py --new test_logos/citroen.jpg --collections logos --cache --visualize --html

python compare_trademarks_v10.py \
    --new new.jpg --collections collections \
    --embedding both --cache \
    --html report.html --visualize

python compare_trademarks_v09.py --new live_logos/citroen.jpg --collections logos --html report.html --visualize --cache

python compare_trademarks_v07.py --new live_logos/citroen.jpg --collections logos --html report.html --visualize --cache


python compare_trademarks_v07.py --new logo.jpg --collections collections --html report.html --visualize --cache

# Без флага — только верхний уровень collections/
python compare_trademarks_v07.py --new new.jpg --collections collections

# С флагом — обход всех подпапок
python compare_trademarks_v07.py --new new.jpg --collections collections --recursive