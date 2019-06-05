for((i=0;i<500;i++))
do
    fn=` printf "model_output/run%04d/I211_J211.dat" $i `
    if [ ! -f "${fn}" ]; then echo $i "The file doesn't exist"; fi
done
