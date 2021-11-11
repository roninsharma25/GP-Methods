python3 TA_systems_driver.py --DRIVE_FITNESS_VALUE 0.93853 --DROP_RATE 0.9890899999999999 --EMBRYO_RESISTANCE_RATE 0.93997 --GERMLINE_RESISTANCE_RATE 0.99211 > data/1.part &
wait
cd data
cat *.part > adaptive.csv
rm *.part
