
# usage:
#    ./my_submission_tarb.sh mypaper.tex myrefs.bib mypaper.bbl img_folder

today=$(date +%Y%m%d)
texf=$1 # tex file
bibf=$2 # bib file 
bblf=$3 # bbl file
imgfld=$4 # img folder

	#Cleaning if the folder already exists
[ -d "submission$today" ] && rm -r submission$today submission$today.tgz

mkdir -vp submission$today

cp -r $texf $bibf $bblf $imgfld submission$today

tar zcvf submission$today.tgz submission$today
