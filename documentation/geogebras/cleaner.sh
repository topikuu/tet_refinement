#! /bin/sh

function clean_tikz {
	local filename="$1"
	cat ${filename} \
	| sed -n '/^\\definecolor/,$p' `#Remove prefix (e.g. \begin{document})` \
	| sed '/^\\end{document}/d' `#Remove postfix` \
	| sed 's/width=2pt,dotted/width=1pt,dotted/g' `#lineStyling` \
	| sed 's/width=2pt,dash/width=1pt,dash/g' \
	| sed 's/width=2pt/width=1.5pt/g' \
	| sed 's/circle (2pt)/circle (4pt)/g' `#Midpoint size`\
	| sed 's/\\begin{tikzpicture}\[/\\begin{tikzpicture}\[scale=0.5,/g' `#Scaling` \
	| sed '/^\\clip/d' `#Remove clipping` \
	> cleaned/${filename%%.*}.tikz
}

for f in *.txt;
do
	clean_tikz ${f}
done
