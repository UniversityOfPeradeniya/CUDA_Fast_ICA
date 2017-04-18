rm results.txt
for i in $(seq 2 2 128)
do
	j=1024
	while [ $j -le 524300 ]
	do
	echo "Doing for $i electrodes and $j sample points"
	echo "" >> results.txt
	./out $i $j ../data/sample354_524288.txt ../output/output.txt >> results.txt
	j=$(echo $j*2 | bc);
	done
echo "" >> results.txt
echo "_____________________________________________________________________________________" >> results.txt
echo "" >> results.txt
done
