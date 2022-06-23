all: mbank.pdf

mbank.pdf: mbank.tex mbank.aux bib
	pdflatex -synctex=1 -interaction=nonstopmode "mbank".tex

mbank.aux: mbank.tex
	pdflatex -synctex=1 -interaction=nonstopmode "mbank".tex

bib: mbank.aux biblio.bib
	bibtex "mbank".aux

clean:
	rm -f mbank.bbl mbank.out mbankNotes.bib mbank.aux mbank.log mbank.blg mbank.toc mbank.synctex.gz mbank.pdf
